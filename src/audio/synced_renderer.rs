// ABOUTME: Device-independent core of the synced player: queue, timing and drift correction
// ABOUTME: Renders interleaved f32 for an output whose playback delay the caller reports

use crate::audio::gain::{GainControl, GainRamp};
use crate::audio::sync_correction::{
    CorrectionPlanner, CorrectionSchedule, EngageGate, OutputDelaySettle, SyncErrorFilter,
};
use crate::audio::{AudioBuffer, AudioFormat};
use crate::error::Error;
use crate::log_sampling::should_log_sample;
use crate::sync::ClockSync;
use dasp_sample::Sample;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Callback for post-processing audio samples before output.
///
/// Receives `&mut [f32]` (interleaved, after gain is applied).
///
/// The callback is invoked on **every** audio callback, including during
/// pre-start silence when the buffer is all zeros. This allows consumers
/// (e.g. VU meters) to observe the silence rather than missing callbacks.
///
/// # Thread Safety
///
/// This closure runs on the **audio callback thread**. It must:
/// - Not block (no locks, I/O, or sleeping)
/// - Not allocate (no `Vec::push`, `Box::new`, etc.)
/// - Not panic (would abort the audio thread)
///
/// # Why `Box<dyn>`?
///
/// Using dynamic dispatch (`Box<dyn FnMut>`) keeps [`SyncedRenderer`] (and so
/// `SyncedPlayer`) a concrete, non-generic type. This simplifies storage,
/// trait object compatibility, and downstream usage at the cost of one vtable
/// indirect call per audio callback (~1 ns vs the ~200 us callback budget).
pub type ProcessCallback = Box<dyn FnMut(&mut [f32]) + Send + 'static>;

/// Maximum static delay in milliseconds. The Sendspin protocol defines
/// `static_delay_ms` over 0–5000; larger values are clamped to this.
pub const MAX_STATIC_DELAY_MS: u16 = 5000;

/// Convert a protocol `static_delay_ms` value to microseconds, clamping to
/// [`MAX_STATIC_DELAY_MS`]. Out-of-range values are clamped rather than
/// rejected so a sloppy server can't disable playback timing entirely.
const fn static_delay_ms_to_us(delay_ms: u16) -> u64 {
    let clamped = if delay_ms > MAX_STATIC_DELAY_MS {
        MAX_STATIC_DELAY_MS
    } else {
        delay_ms
    };
    clamped as u64 * 1_000
}

struct PlaybackQueue {
    queue: VecDeque<AudioBuffer>,
    current: Option<AudioBuffer>,
    index: usize,
    /// Current playback position in **server-time microseconds**. Periodically
    /// reanchored to the server's clock during clock-sync correction, so this
    /// represents "what server timestamp is playing right now", not how much
    /// audio content has been consumed.
    cursor_us: i64,
    cursor_remainder: i64,
    initialized: bool,
    generation: u64,
    force_reanchor: bool,
    /// Buffers enqueued in this generation; the sampling key for the enqueue
    /// trace line. Reset by `clear()`.
    enqueue_count: u64,
}

impl PlaybackQueue {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            current: None,
            index: 0,
            cursor_us: 0,
            cursor_remainder: 0,
            initialized: false,
            generation: 0,
            force_reanchor: true,
            enqueue_count: 0,
        }
    }

    fn clear(&mut self) {
        self.queue.clear();
        self.current = None;
        self.index = 0;
        self.cursor_us = 0;
        self.cursor_remainder = 0;
        self.initialized = false;
        self.generation = self.generation.wrapping_add(1);
        self.force_reanchor = true;
        self.enqueue_count = 0;
    }

    fn push(&mut self, buffer: AudioBuffer) {
        // Initialize the cursor from the first enqueued buffer so the audio
        // callback can see a valid cursor_us before it starts reading. Without
        // this, the callback's pre-start gate can't evaluate timestamps and
        // outputs silence indefinitely. Use the minimum timestamp seen so far
        // since buffers may arrive out of order.
        if !self.initialized {
            self.cursor_us = buffer.timestamp;
            self.cursor_remainder = 0;
            self.initialized = true;
        }

        // When the server rebases its timeline backward (e.g. after event loop
        // starvation), new chunks arrive with timestamps that overlap chunks
        // already in the queue. Remove all overlapping buffers to prevent
        // duplicate audio that causes audible stuttering (~500ms of audio
        // played twice). The server will send fresh buffers for any gaps.
        // This works for any chunk size (the sendspin spec allows arbitrarily
        // small chunks), unlike the previous fixed-threshold approach.
        //
        // The overlap must be at least one frame to count. A sub-frame
        // "overlap" cannot contain a duplicated sample — it is timestamp
        // rounding, not a rebase: at rates where chunks are not a whole
        // number of microseconds (44.1kHz: 1102 frames = 24988.66µs), the
        // server's floor-based timestamp grid steps 24988µs while we measure
        // the buffer as 24989µs, landing a third of all chunks 1µs "inside"
        // their predecessor. Evicting on those phantom overlaps silently
        // discarded ~34% of all 44.1kHz audio (heard as continuous popping).
        let rate = i64::from(buffer.format.sample_rate.max(1));
        let frame_us = (1_000_000 + rate - 1) / rate;
        let new_end = buffer.timestamp + buffer.duration_us();
        self.queue.retain(|b| {
            let existing_end = b.timestamp + b.duration_us();
            let overlap_us = new_end.min(existing_end) - buffer.timestamp.max(b.timestamp);
            overlap_us < frame_us
        });

        let pos = self
            .queue
            .iter()
            .position(|b| b.timestamp > buffer.timestamp);
        if let Some(pos) = pos {
            self.queue.insert(pos, buffer);
        } else {
            self.queue.push_back(buffer);
        }
    }

    fn next_frame(&mut self, channels: usize, sample_rate: u32) -> Option<&[i32]> {
        let needs_buffer = match self.current {
            None => true,
            Some(ref c) => self.index + channels > c.samples.len(),
        };
        if needs_buffer {
            // Drop stale buffers that are entirely before the cursor.
            if self.initialized {
                while let Some(front) = self.queue.front() {
                    if front.timestamp + front.duration_us() < self.cursor_us {
                        let _ = self.queue.pop_front();
                        continue;
                    }
                    break;
                }
            }

            // Pop buffers until we find one with remaining samples past the
            // cursor, or the queue is empty.
            loop {
                self.current = self.queue.pop_front();
                self.index = 0;

                // Skip past samples that are behind the cursor. This handles
                // buffers that partially overlap with the current playback
                // position, e.g. from backward timestamp jumps during server
                // timeline rebases. Without this, playing from the start of
                // such a buffer repeats audio the cursor has already passed,
                // causing an audible stutter.
                if self.initialized {
                    if let Some(ref current) = self.current {
                        if current.timestamp < self.cursor_us {
                            let skip_us = self.cursor_us - current.timestamp;
                            let skip_frames =
                                (skip_us.saturating_mul(sample_rate as i64) / 1_000_000) as usize;
                            if skip_frames > 0 {
                                self.index = skip_frames
                                    .saturating_mul(channels)
                                    .min(current.samples.len());
                            }
                        }
                    }
                }

                // If the skip consumed the entire buffer (or left fewer
                // samples than one frame), discard it and try the next one.
                match self.current {
                    Some(ref c) if self.index + channels > c.samples.len() => {
                        self.current = None;
                        if self.queue.is_empty() {
                            break;
                        }
                        continue;
                    }
                    _ => break,
                }
            }
        }

        if !self.initialized {
            if let Some(current) = self.current.as_ref() {
                self.cursor_us = current.timestamp;
                self.cursor_remainder = 0;
                self.initialized = true;
            }
        }

        // Bail before advancing cursor/index when the queue is empty.
        // Without this the cursor races ahead during underruns, causing
        // the stale-buffer-dropping logic to discard valid buffers when
        // audio resumes.
        self.current.as_ref()?;

        let start = self.index;
        let end = self.index + channels;
        self.index = end;
        self.advance_cursor(sample_rate);

        Some(&self.current.as_ref()?.samples[start..end])
    }

    fn advance_cursor(&mut self, sample_rate: u32) {
        self.cursor_remainder += 1_000_000;
        let advance = self.cursor_remainder / sample_rate as i64;
        self.cursor_remainder %= sample_rate as i64;
        self.cursor_us += advance;
    }

    fn first_playable_cursor_at_or_after(&self, server_time_us: i64) -> Option<i64> {
        if let Some(buffer) = self.current.as_ref() {
            let remaining_start = buffer.timestamp.max(self.cursor_us);
            if buffer.timestamp + buffer.duration_us() > server_time_us.max(remaining_start) {
                return Some(remaining_start.max(server_time_us));
            }
        }

        for buffer in &self.queue {
            if buffer.timestamp + buffer.duration_us() > server_time_us {
                return Some(buffer.timestamp.max(server_time_us));
            }
        }

        None
    }

    fn queued_frames(&self, channels: usize) -> usize {
        let current_frames = self.current.as_ref().map_or(0, |current| {
            current.samples.len().saturating_sub(self.index) / channels
        });
        let queued_frames = self
            .queue
            .iter()
            .map(|buffer| buffer.samples.len() / channels)
            .sum::<usize>();
        current_frames + queued_frames
    }

    fn queued_duration_us(&self, channels: usize, sample_rate: u32) -> u64 {
        self.queued_frames(channels) as u64 * 1_000_000 / sample_rate.max(1) as u64
    }

    fn buffer_count(&self) -> usize {
        self.queue.len() + usize::from(self.current.is_some())
    }
}

/// Microseconds as fractional milliseconds, for log formatting.
fn us_to_ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

/// Queue depth below which the edge-triggered "queue low" debug line fires.
const QUEUE_LOW_WATER_US: u64 = 100_000;

/// Upper bound on how long after playback starts the output delay is given to
/// settle before corrections are allowed regardless (see
/// [`OutputDelaySettle`]). A backend whose delay never settles must not
/// disable drift correction for the whole stream.
const OUTPUT_DELAY_SETTLE_MAX: Duration = Duration::from_secs(10);

/// How long after playback starts speed corrections are limited to
/// [`STARTUP_MAX_SPEED_CORRECTION`]. Whatever residue of the start-up
/// transient outlasts the settle grace period must not be corrected at the
/// full rate, which is audible as pitch.
const STARTUP_GENTLE_WINDOW: Duration = Duration::from_secs(20);

/// Speed-correction cap during [`STARTUP_GENTLE_WINDOW`]: 0.5%, under 9 cents.
const STARTUP_MAX_SPEED_CORRECTION: f64 = 0.005;

/// Errors at or above this are re-anchored in [`CorrectionMode::Smooth`]:
/// 20ms. Slewing more than that at the smooth speed cap would take seconds.
const SMOOTH_REANCHOR_THRESHOLD_US: i64 = 20_000;

/// Speed-correction cap in [`CorrectionMode::Smooth`], during and after
/// startup: 0.5%, under 9 cents.
const SMOOTH_MAX_SPEED_CORRECTION: f64 = 0.005;

/// Observe delay before planning each callback; call `start` when audio begins.
/// Observations may precede playback, but startup deadlines begin at `start`.
struct StartupCorrectionState {
    planner: CorrectionPlanner,
    startup_planner: CorrectionPlanner,
    error_filter: SyncErrorFilter,
    engage_gate: EngageGate,
    delay_settle: OutputDelaySettle,
    started_at: Option<Instant>,
    /// Padding noise is one-sided: anchoring to a high sample bakes that error
    /// into the timeline. Use the generation's floor, replaced on settling.
    min_playback_delta_us: u64,
}

struct CorrectionDecision {
    error_us: i64,
    ungated: CorrectionSchedule,
    gated: CorrectionSchedule,
    schedule: CorrectionSchedule,
    output_delay_transient: bool,
    clock_warmup_suppressed: bool,
}

impl StartupCorrectionState {
    fn new(mode: CorrectionMode) -> Self {
        let (planner, startup_planner) = match mode {
            CorrectionMode::DropInsert => (
                CorrectionPlanner::new(),
                CorrectionPlanner::with_max_speed_correction(STARTUP_MAX_SPEED_CORRECTION),
            ),
            CorrectionMode::Smooth => {
                let planner = CorrectionPlanner::with_limits(
                    SMOOTH_MAX_SPEED_CORRECTION,
                    SMOOTH_REANCHOR_THRESHOLD_US,
                );
                (planner, planner)
            }
        };
        Self {
            planner,
            startup_planner,
            error_filter: SyncErrorFilter::new(),
            engage_gate: EngageGate::new(),
            delay_settle: OutputDelaySettle::new(),
            started_at: None,
            min_playback_delta_us: u64::MAX,
        }
    }

    /// Forget all state except the planners.
    fn reset(&mut self) {
        *self = Self {
            planner: self.planner,
            startup_planner: self.startup_planner,
            ..Self::new(CorrectionMode::default())
        };
    }

    fn reset_measurements(&mut self) {
        self.error_filter.reset();
        self.engage_gate.reset();
    }

    fn start(&mut self, now: Instant) {
        self.started_at = Some(now);
    }

    fn observe_output_delay(&mut self, now: Instant, delta_us: u64) -> Option<u64> {
        self.min_playback_delta_us = self.min_playback_delta_us.min(delta_us);
        let was_settled = self.delay_settle.is_settled();
        self.delay_settle.update(now, delta_us);
        let floor = self
            .delay_settle
            .settled_floor_us()
            .filter(|_| !was_settled);
        if let Some(floor) = floor {
            // A stable interval replaces the transient floor and its error history.
            self.min_playback_delta_us = floor;
            self.reset_measurements();
        }
        floor
    }

    fn plan(
        &mut self,
        now: Instant,
        raw_error_us: i64,
        sample_rate: u32,
        currently_correcting: bool,
        clock_ready: bool,
    ) -> CorrectionDecision {
        let error_us = self.error_filter.update(raw_error_us);
        let elapsed = self.started_at.map(|at| now.duration_since(at));
        let planner = if elapsed.is_some_and(|age| age < STARTUP_GENTLE_WINDOW) {
            &self.startup_planner
        } else {
            &self.planner
        };
        let ungated = planner.plan(error_us, sample_rate, currently_correcting);
        let gated =
            self.engage_gate
                .admit(ungated, currently_correcting, self.error_filter.is_warm());
        let output_delay_transient = !self.delay_settle.is_settled()
            && elapsed.is_some_and(|age| age < OUTPUT_DELAY_SETTLE_MAX);
        let clock_warmup_suppressed = !output_delay_transient && !clock_ready && !gated.reanchor;
        let schedule = if output_delay_transient || clock_warmup_suppressed {
            CorrectionSchedule::default()
        } else {
            gated
        };
        CorrectionDecision {
            error_us,
            ungated,
            gated,
            schedule,
            output_delay_transient,
            clock_warmup_suppressed,
        }
    }
}

#[cfg(test)]
mod startup_correction_tests {
    use super::*;

    fn tick(
        state: &mut StartupCorrectionState,
        now: Instant,
        delta_us: u64,
        error_us: i64,
        schedule: &mut CorrectionSchedule,
        clock_ready: bool,
    ) -> CorrectionDecision {
        state.observe_output_delay(now, delta_us);
        let decision = state.plan(now, error_us, 48_000, schedule.is_correcting(), clock_ready);
        *schedule = decision.schedule;
        decision
    }

    #[test]
    fn cold_start_overshoot_plateau_relaxes_without_correction() {
        let start = Instant::now();
        let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
        state.start(start);
        let mut schedule = CorrectionSchedule::default();
        let mut settled = false;
        let mut saw_transient_gated_correction = false;
        for ms in (0..=7000).step_by(10) {
            let delta = match ms {
                0..=499 => 30_000 + ms * 160,
                500..=1499 => 110_000,
                1500..=3999 => 110_000 - (ms - 1500) * 32,
                _ => 30_000,
            };
            let decision = tick(
                &mut state,
                start + Duration::from_millis(ms),
                delta,
                delta as i64 - 30_000,
                &mut schedule,
                true,
            );
            assert!(!decision.schedule.is_correcting(), "at {ms}ms");
            saw_transient_gated_correction |=
                decision.output_delay_transient && decision.gated.is_correcting();
            if ms < 4000 {
                assert!(!state.delay_settle.is_settled(), "at {ms}ms");
            }
            settled |= state.delay_settle.is_settled();
        }
        assert!(settled);
        assert!(saw_transient_gated_correction);
        assert_eq!(state.min_playback_delta_us, 30_000);
    }

    #[test]
    fn settling_discards_negative_transient_and_requires_fresh_evidence() {
        for real_error in [0, 40_000, -40_000] {
            let start = Instant::now();
            let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
            state.start(start);
            let mut schedule = CorrectionSchedule::default();
            for ms in (0..2500).step_by(10) {
                tick(
                    &mut state,
                    start + Duration::from_millis(ms),
                    50_000,
                    -200_000,
                    &mut schedule,
                    true,
                );
                assert!(!schedule.is_correcting());
            }
            assert!(state.engage_gate.pending_plans() > 0);
            let settled_at = start + Duration::from_millis(2500);
            assert_eq!(state.observe_output_delay(settled_at, 50_000), Some(50_000));
            assert_eq!(state.engage_gate.pending_plans(), 0);
            assert!(!state.error_filter.is_warm());
            let first = state.plan(settled_at, real_error, 48_000, false, true);
            assert_eq!(first.error_us, real_error);
            assert!(!first.schedule.is_correcting());
            let mut first_correction = None;
            for callback in 2..=150 {
                let decision = tick(
                    &mut state,
                    settled_at + Duration::from_millis(callback * 10),
                    50_000,
                    real_error,
                    &mut schedule,
                    true,
                );
                assert_eq!(decision.error_us, real_error);
                if decision.schedule.is_correcting() {
                    first_correction.get_or_insert(callback);
                }
            }
            if real_error == 0 {
                assert_eq!(first_correction, None);
            } else {
                assert_eq!(
                    first_correction,
                    Some(
                        crate::audio::sync_correction::SYNC_ERROR_WINDOW
                            .max(crate::audio::sync_correction::ENGAGE_STREAK_PLANS as usize,)
                            as u64
                    ),
                );
                assert_eq!(schedule.drop_every_n_frames > 0, real_error > 0);
                assert_eq!(schedule.insert_every_n_frames > 0, real_error < 0);
            }
        }
    }

    #[test]
    fn suppression_and_speed_cap_expire_at_exact_deadlines_or_late_callbacks() {
        for late_ms in [0, 137] {
            let start = Instant::now();
            let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
            state.start(start);
            let mut schedule = CorrectionSchedule::default();
            for ms in (0..10_000).step_by(10) {
                let decision = tick(
                    &mut state,
                    start + Duration::from_millis(ms),
                    30_000 + ms * 20,
                    100_000,
                    &mut schedule,
                    true,
                );
                assert!(decision.output_delay_transient);
                assert!(!schedule.is_correcting());
            }
            let before = state.plan(
                start + OUTPUT_DELAY_SETTLE_MAX - Duration::from_nanos(1),
                100_000,
                48_000,
                false,
                true,
            );
            assert!(before.output_delay_transient);
            assert!(!before.schedule.is_correcting());
            let deadline = tick(
                &mut state,
                start + OUTPUT_DELAY_SETTLE_MAX + Duration::from_millis(late_ms),
                240_000,
                100_000,
                &mut schedule,
                true,
            );
            assert!(!deadline.output_delay_transient);
            assert_eq!(schedule.drop_every_n_frames, 200);
            for ms in (10_150..20_000).step_by(10) {
                let decision = tick(
                    &mut state,
                    start + Duration::from_millis(ms),
                    250_000 + (ms - 10_150) * 20,
                    100_000,
                    &mut schedule,
                    true,
                );
                assert!(!decision.output_delay_transient);
                assert_eq!(schedule.drop_every_n_frames, 200);
                assert!(!state.delay_settle.is_settled());
            }
            let before_cap = state.plan(
                start + STARTUP_GENTLE_WINDOW - Duration::from_nanos(1),
                100_000,
                48_000,
                true,
                true,
            );
            assert_eq!(before_cap.schedule.drop_every_n_frames, 200);
            tick(
                &mut state,
                start + STARTUP_GENTLE_WINDOW + Duration::from_millis(late_ms),
                300_000,
                100_000,
                &mut schedule,
                true,
            );
            assert_eq!(schedule.drop_every_n_frames, 25);
        }
    }

    #[test]
    fn smooth_mode_reanchors_at_20ms_and_caps_speed_before_and_after_startup() {
        for after_startup in [false, true] {
            let ungated = |error_us| {
                let start = Instant::now();
                let mut state = StartupCorrectionState::new(CorrectionMode::Smooth);
                // A reset keeps the mode's planners.
                state.reset();
                state.start(start);
                let age = if after_startup {
                    STARTUP_GENTLE_WINDOW + Duration::from_secs(1)
                } else {
                    Duration::from_secs(1)
                };
                state
                    .plan(start + age, error_us, 48_000, true, true)
                    .ungated
            };
            assert!(ungated(20_000).reanchor);
            assert!(ungated(-20_000).reanchor);
            assert_eq!(ungated(19_000).drop_every_n_frames, 200, "0.5% cap");
            assert_eq!(ungated(-19_000).insert_every_n_frames, 200, "0.5% cap");
        }
    }

    #[test]
    fn generation_reset_restarts_floor_filter_gate_and_deadlines() {
        let start = Instant::now();
        let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
        state.start(start);
        let mut schedule = CorrectionSchedule::default();
        for ms in (0..=4000).step_by(10) {
            tick(
                &mut state,
                start + Duration::from_millis(ms),
                10_000,
                -100_000,
                &mut schedule,
                true,
            );
        }
        assert!(state.delay_settle.is_settled());
        assert!(schedule.is_correcting());
        state.engage_gate.admit(
            CorrectionSchedule {
                drop_every_n_frames: 200,
                ..Default::default()
            },
            false,
            false,
        );
        assert_eq!(state.engage_gate.pending_plans(), 1);
        state.reset();
        assert_eq!(state.engage_gate.pending_plans(), 0);
        schedule = CorrectionSchedule::default();
        assert_eq!(state.started_at, None);
        assert_eq!(state.min_playback_delta_us, u64::MAX);
        assert!(!state.delay_settle.is_settled());
        assert!(!state.error_filter.is_warm());
        let restart = start + Duration::from_secs(30);
        state.start(restart);
        let first = tick(&mut state, restart, 80_000, 100_000, &mut schedule, true);
        assert_eq!(first.error_us, 100_000);
        assert_eq!(first.gated, CorrectionSchedule::default());
        assert!(first.output_delay_transient);
        assert_eq!(state.min_playback_delta_us, 80_000);
        for ms in (10..=4000).step_by(10) {
            tick(
                &mut state,
                restart + Duration::from_millis(ms),
                80_000,
                100_000,
                &mut schedule,
                true,
            );
        }
        assert_eq!(schedule.drop_every_n_frames, 200);
    }

    #[test]
    fn measurement_reset_preserves_startup_timing_and_delay_floor() {
        let start = Instant::now();
        let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
        state.start(start);
        let mut schedule = CorrectionSchedule::default();
        for ms in (0..=4000).step_by(10) {
            tick(
                &mut state,
                start + Duration::from_millis(ms),
                50_000,
                100_000,
                &mut schedule,
                true,
            );
        }
        assert!(schedule.is_correcting());
        state.engage_gate.admit(
            CorrectionSchedule {
                drop_every_n_frames: 200,
                ..Default::default()
            },
            false,
            false,
        );
        assert_eq!(state.engage_gate.pending_plans(), 1);
        state.reset_measurements();
        assert_eq!(state.engage_gate.pending_plans(), 0);
        assert_eq!(state.started_at, Some(start));
        assert_eq!(state.min_playback_delta_us, 50_000);
        assert!(state.delay_settle.is_settled());
        assert!(!state.error_filter.is_warm());
        let decision = state.plan(
            start + Duration::from_secs(4),
            -100_000,
            48_000,
            false,
            true,
        );
        assert_eq!(decision.error_us, -100_000);
        assert!(!decision.gated.is_correcting());
    }

    #[test]
    fn settled_edges_replace_upward_floor_and_reset_measurements_only_once() {
        let start = Instant::now();
        let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
        state.start(start);
        assert_eq!(state.observe_output_delay(start, 5_000), None);
        let mut transitions = Vec::new();
        for ms in (10..=4000).step_by(10) {
            if let Some(floor) =
                state.observe_output_delay(start + Duration::from_millis(ms), 50_000)
            {
                transitions.push(floor);
            }
        }
        assert_eq!(transitions, vec![50_000]);
        assert_eq!(state.min_playback_delta_us, 50_000);
        for callback in 1..=150 {
            let now = start + Duration::from_millis(4000 + callback * 10);
            assert_eq!(state.observe_output_delay(now, 50_000), None);
            state.plan(now, -100_000, 48_000, false, true);
        }
        assert!(state.error_filter.is_warm());
        assert!(state.engage_gate.pending_plans() > 0);
        let mut saw_unsettled = false;
        let mut second_transition = None;
        for ms in (5510..=9000).step_by(10) {
            let now = start + Duration::from_millis(ms);
            let floor = state.observe_output_delay(now, 90_000);
            saw_unsettled |= !state.delay_settle.is_settled();
            if floor.is_some() {
                second_transition = Some(now);
                assert_eq!(floor, Some(90_000));
                assert!(!state.error_filter.is_warm());
                assert_eq!(state.engage_gate.pending_plans(), 0);
                assert_eq!(state.started_at, Some(start));
                assert_eq!(state.min_playback_delta_us, 90_000);
                break;
            }
        }
        assert!(saw_unsettled);
        let settled_at = second_transition.expect("higher delay settles again");
        for callback in 1..=150 {
            let now = settled_at + Duration::from_millis(callback * 10);
            assert_eq!(state.observe_output_delay(now, 90_000), None);
            let decision = state.plan(now, 100_000, 48_000, false, true);
            assert_eq!(decision.error_us, 100_000);
        }
        assert!(state.error_filter.is_warm());
        assert!(state.engage_gate.pending_plans() > 0);
        assert_eq!(state.started_at, Some(start));
    }

    #[test]
    fn pre_start_observations_settle_delay_but_deadlines_use_actual_start() {
        let observed_at = Instant::now();
        let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
        for ms in (0..=3000).step_by(10) {
            state.observe_output_delay(observed_at + Duration::from_millis(ms), 50_000);
        }
        assert!(state.delay_settle.is_settled());
        assert_eq!(state.started_at, None);
        assert!(!state.error_filter.is_warm());
        assert_eq!(state.engage_gate.pending_plans(), 0);
        let start = observed_at + Duration::from_secs(30);
        state.start(start);
        let mut schedule = CorrectionSchedule::default();
        for ms in (0..10_000).step_by(10) {
            let decision = tick(
                &mut state,
                start + Duration::from_millis(ms),
                50_000 + ms * 20,
                100_000,
                &mut schedule,
                true,
            );
            assert!(decision.output_delay_transient);
            assert!(!schedule.is_correcting());
        }
        let deadline = state.plan(
            start + OUTPUT_DELAY_SETTLE_MAX,
            100_000,
            48_000,
            false,
            true,
        );
        assert!(!deadline.output_delay_transient);
        assert_eq!(deadline.schedule.drop_every_n_frames, 200);
        let before_cap = state.plan(
            start + STARTUP_GENTLE_WINDOW - Duration::from_nanos(1),
            100_000,
            48_000,
            true,
            true,
        );
        assert_eq!(before_cap.schedule.drop_every_n_frames, 200);
        let after_cap = state.plan(start + STARTUP_GENTLE_WINDOW, 100_000, 48_000, true, true);
        assert_eq!(after_cap.schedule.drop_every_n_frames, 25);
    }

    #[test]
    fn clock_readiness_cannot_bypass_output_settling_but_allows_gross_reanchor() {
        let start = Instant::now();
        let mut state = StartupCorrectionState::new(CorrectionMode::DropInsert);
        state.start(start);
        let initial = state.plan(start, 600_000, 48_000, false, true);
        assert!(initial.gated.reanchor);
        assert!(!initial.schedule.is_correcting());
        let mut schedule = CorrectionSchedule::default();
        let mut saw_clock_suppression = false;
        for ms in (0..=4000).step_by(10) {
            let decision = tick(
                &mut state,
                start + Duration::from_millis(ms),
                50_000,
                100_000,
                &mut schedule,
                false,
            );
            assert!(!schedule.is_correcting());
            if decision.gated.is_correcting() && !decision.output_delay_transient {
                assert!(state.error_filter.is_warm());
                assert!(decision.clock_warmup_suppressed);
                saw_clock_suppression = true;
            }
        }
        assert!(state.delay_settle.is_settled());
        assert!(saw_clock_suppression);
        state.reset_measurements();
        assert!(!state.error_filter.is_warm());
        assert_eq!(state.engage_gate.pending_plans(), 0);
        let evidence_needed = crate::audio::sync_correction::SYNC_ERROR_WINDOW
            .max(crate::audio::sync_correction::ENGAGE_STREAK_PLANS as usize);
        for callback in 1..=evidence_needed {
            let decision = tick(
                &mut state,
                start + Duration::from_millis(4000 + callback as u64 * 10),
                50_000,
                -100_000,
                &mut schedule,
                true,
            );
            assert_eq!(decision.error_us, -100_000);
            assert!(!decision.clock_warmup_suppressed);
            assert_eq!(schedule.is_correcting(), callback == evidence_needed);
        }
        assert!(schedule.insert_every_n_frames > 0);
        state.reset_measurements();
        let gross = state.plan(
            start + Duration::from_secs(6),
            600_000,
            48_000,
            false,
            false,
        );
        assert!(gross.schedule.reanchor);
    }
}

/// Queue depth required to log recovery after a low-queue warning. Kept above
/// [`QUEUE_LOW_WATER_US`] so a queue hovering at one boundary cannot flood the
/// log with low/recovered pairs.
const QUEUE_RECOVERED_WATER_US: u64 = 200_000;

/// Diagnostic counters for the audio callback. Logging-only: playback
/// decisions never read these.
///
/// `callbacks` counts for the lifetime of the stream and anchors every log
/// line to one timeline. The rest are per-generation — reset whenever the
/// playback queue generation changes — so each stream start reports its own
/// startup behavior.
#[derive(Default)]
struct CallbackStats {
    /// Data callbacks since the stream was built. Never reset.
    callbacks: u64,
    /// Callbacks that emitted silence (pre-start gate, reanchor wait, early).
    silent_callbacks: u64,
    /// Callbacks that skipped sync because the clock lock was contended.
    sync_lock_misses: u64,
    /// Frames filled with silence because the queue ran dry.
    underrun_frames: u64,
    /// Callbacks that had at least one underrun frame.
    underrun_callbacks: u64,
    /// Length of the current run of underrun callbacks (0 while healthy).
    consecutive_underrun_callbacks: u64,
    /// Schedule updates within the current correction episode; the sampling
    /// key for the correction trace line. Reset when correction disengages.
    correction_updates: u64,
    /// Corrections the planner requested during clock warm-up that were
    /// suppressed; the sampling key for the warm-up trace line.
    warmup_suppressed_corrections: u64,
    /// Corrections the planner requested that the engage gate suppressed
    /// while awaiting a sustained error; the sampling key for its trace line.
    gate_suppressed_corrections: u64,
    /// Corrections the planner requested while the output delay was still
    /// settling after stream start, which were suppressed; the sampling key
    /// for its trace line.
    delay_transient_suppressed_corrections: u64,
    /// Correction episodes started (idle -> correcting transitions, including
    /// reanchor engagements). Mirrors the "Sync correction engaged" debug
    /// line 1:1 so the generation summary can answer whether the corrector
    /// ever fired, even when debug logging was off during playback.
    correction_engagements: u64,
    /// Whether the queue was below [`QUEUE_LOW_WATER_US`] at the last render.
    /// Drives the edge-triggered low/recovered debug lines.
    queue_low: bool,
}

impl CallbackStats {
    /// Reset per-generation counters, keeping the lifetime callback count.
    fn reset_for_generation(&mut self) {
        *self = Self {
            callbacks: self.callbacks,
            ..Self::default()
        };
    }
}

/// How a [`SyncedRenderer`] corrects drift between playback and the server
/// clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorrectionMode {
    /// Drop or repeat single frames at a cadence for errors under 500ms, at
    /// up to 4% of real time (0.5% for the first 20 seconds of playback),
    /// and jump the playback position for larger ones. Each dropped or
    /// repeated frame is a small discontinuity, which at high cadences can
    /// be heard as clicks or garbled audio.
    #[default]
    DropInsert,
    /// Jump the playback position for errors of 20ms or more, and correct
    /// smaller ones by reading the queue at a slightly different rate with
    /// linear interpolation, at up to 0.5% of real time. Large errors are
    /// heard as a brief silence or skip rather than a long stretch of
    /// altered audio; small ones as a pitch shift under 9 cents.
    Smooth,
}

/// Configuration for [`SyncedRenderer`] construction.
#[derive(Debug, Clone)]
pub struct SyncedRendererConfig {
    /// Initial playback volume, 0-100.
    pub volume: u8,
    /// Initial mute state.
    pub muted: bool,
    /// How drift is corrected. Defaults to [`CorrectionMode::DropInsert`].
    pub correction: CorrectionMode,
}

impl SyncedRendererConfig {
    /// Create a config with common playback defaults.
    pub fn new() -> Self {
        Self {
            volume: 100,
            muted: false,
            correction: CorrectionMode::default(),
        }
    }
}

impl Default for SyncedRendererConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Control side of a [`SyncedRenderer`]: queues decoded audio for it and
/// sets its volume, mute and static delay.
///
/// It shares its state with the renderer, so it can be used from any thread
/// while the output's thread renders. Cloning is cheap and every clone
/// controls the same renderer.
#[derive(Clone)]
pub struct SyncedSource {
    format: AudioFormat,
    queue: Arc<Mutex<PlaybackQueue>>,
    gain: GainControl,
    /// Shared with the renderer. See [`SyncedSource::set_static_delay`].
    static_delay_us: Arc<AtomicU64>,
}

impl SyncedSource {
    /// Enqueue a decoded buffer for playback.
    ///
    /// Scheduling uses `buffer.timestamp` (server time in microseconds) for
    /// drift-corrected playback.
    pub fn enqueue(&self, buffer: AudioBuffer) {
        let buffer_timestamp = buffer.timestamp;
        let buffer_duration_us = buffer.duration_us();
        let channels = self.format.channels as usize;
        let sample_rate = self.format.sample_rate;

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        // The O(buffers) depth walk runs only for sampled, trace-enabled
        // enqueues.
        let trace_fields = {
            let mut queue = self.queue.lock();
            queue.push(buffer);
            queue.enqueue_count += 1;
            if log::log_enabled!(log::Level::Trace) && should_log_sample(queue.enqueue_count) {
                Some((
                    queue.enqueue_count,
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                    queue.cursor_us,
                    queue.generation,
                ))
            } else {
                None
            }
        };

        if let Some((enqueue_count, queued_us, buffers, cursor_us, generation)) = trace_fields {
            log::trace!(
                "Audio buffer enqueued: enqueue={}, ts={}µs, duration={:.1}ms, queued={:.1}ms, buffers={}, cursor={}µs, generation={}",
                enqueue_count,
                buffer_timestamp,
                buffer_duration_us as f64 / 1000.0,
                us_to_ms(queued_us),
                buffers,
                cursor_us,
                generation,
            );
        }
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        let channels = self.format.channels as usize;
        let sample_rate = self.format.sample_rate;

        // Snapshot log fields under the lock but log after dropping it: the
        // audio callback contends on this lock, and logging can block on I/O.
        let debug_fields = {
            let mut queue = self.queue.lock();
            let fields = log::log_enabled!(log::Level::Debug).then(|| {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                    queue.cursor_us,
                    queue.generation,
                )
            });
            queue.clear();
            fields
        };

        if let Some((queued_us, buffers, cursor_us, generation)) = debug_fields {
            log::debug!(
                "Cleared playback queue: queued={:.1}ms, buffers={}, cursor={}µs, generation={}",
                us_to_ms(queued_us),
                buffers,
                cursor_us,
                generation,
            );
        }
    }

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        &self.format
    }

    /// Get a reference to the volume/mute control.
    ///
    /// Call `.clone()` if you need an owned handle to share across threads
    /// (cloning is cheap — single `Arc` increment, no data copy).
    pub fn gain_control(&self) -> &GainControl {
        &self.gain
    }

    // -- Volume/mute convenience methods --
    //
    // These promote the most common operations for ergonomics in simple
    // use-cases. For full control, use `gain_control()` directly.

    /// Current volume as 0-100.
    pub fn volume(&self) -> u8 {
        self.gain.volume()
    }

    /// Whether playback is currently muted.
    pub fn is_muted(&self) -> bool {
        self.gain.is_muted()
    }

    /// Set playback volume (0-100).
    pub fn set_volume(&self, volume: u8) {
        self.gain.set_volume(volume);
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        self.gain.set_mute(muted);
    }

    /// Set the static playback delay in milliseconds (0–[`MAX_STATIC_DELAY_MS`]).
    ///
    /// Compensates for external speaker/amplifier latency: the server pre-sends
    /// audio by this amount, so the player shifts each sample's emission earlier
    /// by the same delay to keep alignment correct. Values above the maximum are
    /// clamped. Takes effect on the next audio callback.
    ///
    /// A delay change is an intentional local timing offset, not clock drift.
    /// Request a one-shot reanchor so the audio callback either skips forward or
    /// waits for the new target time instead of feeding the delay delta through
    /// pitch-shifting drift correction.
    pub fn set_static_delay(&self, delay_ms: u16) {
        self.static_delay_us
            .store(static_delay_ms_to_us(delay_ms), Ordering::Relaxed);

        let mut queue = self.queue.lock();
        if queue.initialized {
            queue.force_reanchor = true;
        }
    }

    /// Current static delay in milliseconds.
    pub fn static_delay_ms(&self) -> u16 {
        (self.static_delay_us.load(Ordering::Relaxed) / 1_000) as u16
    }
}

/// Gain and post-processing applied to every rendered buffer, so what the
/// renderer hands back is what the output plays.
struct OutputStage {
    channels: usize,
    gain_control: GainControl,
    gain_ramp: GainRamp,
    process_callback: Option<ProcessCallback>,
}

impl OutputStage {
    /// Fill `out` with silence. The gain ramp advances even while silent so
    /// the first real audio resumes at the target gain with no fade-in.
    fn emit_silence(&mut self, out: &mut [f32]) {
        let target = self.gain_control.gain();
        self.gain_ramp.advance(out.len() / self.channels, target);
        out.fill(0.0);
        self.process(out);
    }

    /// Apply gain with per-frame ramping, then the process callback.
    fn emit(&mut self, out: &mut [f32]) {
        let target = self.gain_control.gain();
        self.gain_ramp.apply(out, self.channels, target);
        self.process(out);
    }

    fn process(&mut self, out: &mut [f32]) {
        if let Some(ref mut cb) = self.process_callback {
            cb(out);
        }
    }
}

/// Input frames consumed per output frame under `schedule`: dropping one
/// frame in N reads N+1 frames per N output, inserting one reads N-1.
fn read_step(schedule: &CorrectionSchedule) -> f64 {
    if schedule.drop_every_n_frames > 0 {
        1.0 + 1.0 / f64::from(schedule.drop_every_n_frames)
    } else if schedule.insert_every_n_frames > 0 {
        1.0 - 1.0 / f64::from(schedule.insert_every_n_frames)
    } else {
        1.0
    }
}

/// Reads the queue at a fractional rate for [`CorrectionMode::Smooth`],
/// interpolating linearly between consecutive input frames.
///
/// The frame before the read position is the renderer's `last_frame`: the
/// last frame taken from the queue, or the one before the lookahead frame
/// held here. Every input frame goes through [`PlaybackQueue::next_frame`],
/// so the cursor advances exactly one frame per input frame, as it does when
/// reading frame by frame; while interpolating it runs up to one frame ahead
/// for the lookahead.
struct FractionalReader {
    /// The input frame after `last_frame`, valid while `has_next`.
    next: Vec<i32>,
    has_next: bool,
    /// Read position past `last_frame`, in input frames, in `0.0..1.0`.
    phase: f64,
    /// The step of the last correction, kept to finish it on.
    step: f64,
}

impl FractionalReader {
    fn new(channels: usize) -> Self {
        Self {
            next: vec![i32::EQUILIBRIUM; channels],
            has_next: false,
            phase: 0.0,
            step: 1.0,
        }
    }

    /// Drop the lookahead frame and the fractional position, e.g. after the
    /// cursor jumped.
    fn reset(&mut self) {
        self.has_next = false;
        self.phase = 0.0;
        self.step = 1.0;
    }

    /// Write one output frame to `out`, advancing the read position by
    /// `step` input frames. Returns false on an underrun, with `out` silent.
    fn read_frame(
        &mut self,
        step: f64,
        last_frame: &mut [i32],
        queue: &mut PlaybackQueue,
        channels: usize,
        sample_rate: u32,
        out: &mut [f32],
    ) -> bool {
        // When the correction ends between two input frames, it is finished
        // at its own step until the position next lands within one step's
        // excess (`|step - 1|`) of a whole frame, and snapped back to that
        // frame there. A correction moves the position by that excess each
        // frame, so this takes at most `1 / |step - 1|` frames, and the snap
        // is smaller than the correction's own per-frame change. Snapping
        // straight away could instead repeat or skip up to a whole frame.
        let finishing = step == 1.0 && self.phase > 0.0;
        let step = if finishing { self.step } else { step };

        if step == 1.0 {
            // Not correcting, on a whole frame: read frame by frame, the
            // held lookahead frame first.
            if self.has_next {
                self.has_next = false;
                last_frame.copy_from_slice(&self.next);
            } else if let Some(frame) = queue.next_frame(channels, sample_rate) {
                last_frame.copy_from_slice(frame);
            } else {
                out.fill(0.0);
                return false;
            }
            for (out, sample) in out.iter_mut().zip(last_frame.iter()) {
                *out = f32::from_sample(*sample);
            }
            return true;
        }

        if !finishing {
            self.step = step;
        }
        self.phase += step;
        while self.phase >= 1.0 {
            if self.has_next {
                self.has_next = false;
                last_frame.copy_from_slice(&self.next);
            } else if let Some(frame) = queue.next_frame(channels, sample_rate) {
                last_frame.copy_from_slice(frame);
            } else {
                // Resume from the last frame read once audio returns.
                self.phase = 0.0;
                out.fill(0.0);
                return false;
            }
            self.phase -= 1.0;
        }
        if finishing && self.phase < (step - 1.0).abs() {
            self.phase = 0.0;
        }
        if self.phase > 0.0 && !self.has_next {
            if let Some(frame) = queue.next_frame(channels, sample_rate) {
                self.next.copy_from_slice(frame);
                self.has_next = true;
            } else {
                self.phase = 0.0;
                out.fill(0.0);
                return false;
            }
        }
        if self.has_next {
            for ((out, from), to) in out.iter_mut().zip(last_frame.iter()).zip(&self.next) {
                let from = f64::from_sample(*from);
                let to = f64::from_sample(*to);
                *out = (from + (to - from) * self.phase) as f32;
            }
        } else {
            for (out, sample) in out.iter_mut().zip(last_frame.iter()) {
                *out = f32::from_sample(*sample);
            }
        }
        true
    }
}

/// Synchronized playback for an output the caller drives.
///
/// `SyncedPlayer` (the `player` feature) opens a cpal stream and runs
/// a renderer in its data callback. Use a renderer directly when the output
/// is something else, such as a platform audio API's write loop or render
/// callback: call [`render`](Self::render) each time the output needs audio,
/// with how long it will be until the first frame of that buffer is heard.
///
/// The renderer pulls frames from the queue its [`SyncedSource`] feeds,
/// aligned to the server clock: it holds silence until the first queued
/// audio is due, then corrects drift as its [`CorrectionMode`] says, and
/// applies gain and the optional [`ProcessCallback`]. What it writes is final.
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use std::time::Duration;
/// # use parking_lot::Mutex;
/// # use sendspin::audio::{AudioFormat, Codec, SyncedRenderer, SyncedRendererConfig};
/// # use sendspin::sync::ClockSync;
/// # use sendspin::DefaultClock;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let format = AudioFormat {
///     codec: Codec::Pcm,
///     sample_rate: 48_000,
///     channels: 2,
///     bit_depth: 24,
///     codec_header: None,
/// };
/// let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));
/// let (source, mut renderer) =
///     SyncedRenderer::new(format, clock_sync, SyncedRendererConfig::new())?;
///
/// // Decoded audio goes to the source, from any thread...
/// source.set_volume(80);
///
/// // ...and the output's thread renders 10ms at a time, here heard 20ms
/// // after the call.
/// let mut out = vec![0.0f32; 480 * 2];
/// renderer.render(&mut out, Duration::from_millis(20));
/// # Ok(())
/// # }
/// ```
pub struct SyncedRenderer {
    channels: usize,
    sample_rate: u32,
    queue: Arc<Mutex<PlaybackQueue>>,
    clock_sync: Arc<Mutex<ClockSync>>,
    static_delay_us: Arc<AtomicU64>,
    output: OutputStage,
    correction: StartupCorrectionState,
    correction_mode: CorrectionMode,
    last_frame: Vec<i32>,
    reader: FractionalReader,
    schedule: CorrectionSchedule,
    insert_counter: u32,
    drop_counter: u32,
    started: bool,
    handoff_warned: bool,
    sync_settle_logged: bool,
    last_callback_instant: Option<Instant>,
    last_playback_delta_us: Option<u64>,
    last_generation: u64,
    stats: CallbackStats,
}

impl SyncedRenderer {
    /// Create a renderer and the [`SyncedSource`] that feeds and controls it.
    ///
    /// The renderer starts at the configured volume and mute state, with no
    /// ramp from a default value. Fails if `format` has no channels.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedRendererConfig,
    ) -> Result<(SyncedSource, Self), Error> {
        Self::build(format, clock_sync, config, None)
    }

    /// Create a renderer with a process callback for post-gain audio
    /// processing, and the [`SyncedSource`] that feeds and controls it.
    ///
    /// The callback receives samples **after** gain/mute processing has been
    /// applied. See [`ProcessCallback`] for thread-safety requirements.
    pub fn with_process_callback(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedRendererConfig,
        callback: ProcessCallback,
    ) -> Result<(SyncedSource, Self), Error> {
        Self::build(format, clock_sync, config, Some(callback))
    }

    pub(crate) fn build(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedRendererConfig,
        process_callback: Option<ProcessCallback>,
    ) -> Result<(SyncedSource, Self), Error> {
        if format.channels == 0 {
            return Err(Error::Output("channels must be > 0".to_string()));
        }
        let channels = format.channels as usize;
        let sample_rate = format.sample_rate;
        let queue = Arc::new(Mutex::new(PlaybackQueue::new()));
        let gain = GainControl::new(config.volume, config.muted);
        let static_delay_us = Arc::new(AtomicU64::new(0));

        let renderer = Self {
            channels,
            sample_rate,
            queue: Arc::clone(&queue),
            clock_sync,
            static_delay_us: Arc::clone(&static_delay_us),
            output: OutputStage {
                channels,
                gain_ramp: GainRamp::new(sample_rate, gain.gain()),
                gain_control: gain.clone(),
                process_callback,
            },
            correction: StartupCorrectionState::new(config.correction),
            correction_mode: config.correction,
            last_frame: vec![i32::EQUILIBRIUM; channels],
            reader: FractionalReader::new(channels),
            schedule: CorrectionSchedule::default(),
            insert_counter: 0,
            drop_counter: 0,
            started: false,
            handoff_warned: false,
            sync_settle_logged: false,
            last_callback_instant: None,
            last_playback_delta_us: None,
            last_generation: 0,
            stats: CallbackStats::default(),
        };
        let source = SyncedSource {
            format,
            queue,
            gain,
            static_delay_us,
        };
        Ok((source, renderer))
    }

    /// Fill `out` with the audio due when its first frame is heard,
    /// `output_delay` from now.
    ///
    /// `out` is interleaved in the format's channel order and should hold a
    /// whole number of frames; a trailing partial frame is filled with
    /// silence. Samples are in -1.0..=1.0 with gain and the process callback
    /// applied.
    ///
    /// `output_delay` is the time from this call until the first frame of
    /// `out` reaches the listener: whatever the output has already queued
    /// ahead of this buffer plus its own latency, as the platform reports it.
    /// The renderer takes the current time itself, so the delay should be
    /// measured as close to the call as practical. Report it consistently:
    /// a delay that is off by a constant shifts playback against other
    /// players by the same amount, and jitter in it is treated as drift.
    ///
    /// Call this from the output's thread, once per buffer it asks for. It
    /// does not allocate. It shares the queue lock with the source, which
    /// holds it only to insert or clear buffers, and it skips clock
    /// synchronization for a buffer rather than wait for the clock lock.
    pub fn render(&mut self, out: &mut [f32], output_delay: Duration) {
        let channels = self.channels;
        let sample_rate = self.sample_rate;

        // Snapshot the level checks once per callback. At info level
        // these two loads are the only per-callback logging cost.
        let debug_logging = log::log_enabled!(log::Level::Debug);
        let trace_logging = log::log_enabled!(log::Level::Trace);

        self.stats.callbacks += 1;
        let frames = out.len() / channels;
        // Only whole frames are rendered; a trailing partial frame stays silent.
        let (out, partial_frame) = out.split_at_mut(frames * channels);
        partial_frame.fill(0.0);

        // Read queue timing state together. The generation is
        // rechecked before consuming force_reanchor so a clear()
        // racing with this callback cannot clear the next startup's
        // one-shot handoff.
        let (generation, cursor_us, force_reanchor, queued_us, queued_buffers) = {
            let queue = self.queue.lock();
            let cursor = if queue.initialized {
                Some(queue.cursor_us)
            } else {
                None
            };
            // Queue depth costs a walk over every queued buffer,
            // so only measure it when a log line below can print
            // it.
            let (queued_us, queued_buffers) = if debug_logging {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                )
            } else {
                (0, 0)
            };
            (
                queue.generation,
                cursor,
                queue.force_reanchor,
                queued_us,
                queued_buffers,
            )
        };
        if generation != self.last_generation {
            log::debug!(
                "Playback queue generation changed: {} -> {}, queued={:.1}ms, buffers={}, callbacks={}, silent_callbacks={}, underrun_callbacks={}, underrun_frames={}, sync_lock_misses={}, correction_engagements={}",
                self.last_generation,
                generation,
                us_to_ms(queued_us),
                queued_buffers,
                self.stats.callbacks,
                self.stats.silent_callbacks,
                self.stats.underrun_callbacks,
                self.stats.underrun_frames,
                self.stats.sync_lock_misses,
                self.stats.correction_engagements,
            );
            self.last_generation = generation;
            self.started = false;
            self.schedule = CorrectionSchedule::default();
            self.insert_counter = 0;
            self.drop_counter = 0;
            self.reader.reset();
            self.correction.reset();
            self.stats.reset_for_generation();
            for sample in self.last_frame.iter_mut() {
                *sample = i32::EQUILIBRIUM;
            }
            self.handoff_warned = false;
        }

        let callback_instant = Instant::now();
        let playback_instant = callback_instant + output_delay;

        // Both values are normally steady, so a step in either
        // explains a sync-error step: a callback gap means this
        // thread stalled; a playback-delta shift means the OS
        // moved the presentation timeline.
        let playback_delta_us = output_delay.as_micros() as u64;
        if let Some(last) = self.last_callback_instant {
            let gap_us = callback_instant.duration_since(last).as_micros() as u64;
            let period_us = frames as u64 * 1_000_000 / u64::from(sample_rate.max(1));
            if gap_us >= 2 * period_us {
                log::debug!(
                    "Audio callback gap: {:.1}ms since previous (period ~{:.1}ms), callback={}, generation={}",
                    us_to_ms(gap_us),
                    us_to_ms(period_us),
                    self.stats.callbacks,
                    generation,
                );
            }
        }
        self.last_callback_instant = Some(callback_instant);
        if let Some(last) = self.last_playback_delta_us {
            if playback_delta_us.abs_diff(last) > 1_000 {
                log::debug!(
                    "Output timeline shifted: playback_delta {:.1}ms -> {:.1}ms, callback={}, generation={}",
                    us_to_ms(last),
                    us_to_ms(playback_delta_us),
                    self.stats.callbacks,
                    generation,
                );
            }
        }
        self.last_playback_delta_us = Some(playback_delta_us);
        // Observe delay even while emitting pre-start silence; start the grace
        // periods only at actual playback, then plan corrections below.
        if let Some(floor_us) = self
            .correction
            .observe_output_delay(callback_instant, playback_delta_us)
        {
            log::debug!(
                "Output delay settled at {:.1}ms; corrections enabled: callback={}, suppressed_during_transient={}, generation={}",
                us_to_ms(floor_us),
                self.stats.callbacks,
                self.stats.delay_transient_suppressed_corrections,
                generation,
            );
        }

        // try_lock: skip sync if contended rather than blocking
        // the audio thread. force_reanchor is sticky in the
        // queue, so it will be retried on the next callback.
        let sync = self.clock_sync.try_lock();
        if cursor_us.is_some() && sync.is_none() {
            // Count lock contention only once playback has an
            // initialized cursor; before that there is no timeline
            // position to synchronize yet.
            self.stats.sync_lock_misses += 1;
            if trace_logging && should_log_sample(self.stats.sync_lock_misses) {
                log::trace!(
                    "Audio callback skipped sync: clock lock contended, callback={}, sync_lock_miss={}, queued={:.1}ms, buffers={}, started={}",
                    self.stats.callbacks,
                    self.stats.sync_lock_misses,
                    us_to_ms(queued_us),
                    queued_buffers,
                    self.started,
                );
            }
        }
        if let (Some(cursor_us), Some(sync)) = (cursor_us, sync) {
            // Emit each sample `delay` earlier so downstream
            // (amp/speaker) latency lands it on time. The reanchor
            // below adds the same delay in the local→server
            // direction; the two signs must stay in step or the
            // planner chases a phantom error every callback.
            let delay_us = self.static_delay_us.load(Ordering::Relaxed);
            let mut effective_cursor_us = cursor_us;
            let sync_settled = sync.is_settled();
            if sync_settled && !self.sync_settle_logged {
                self.sync_settle_logged = true;
                // Warm-up measurements track the converging clock
                // estimate, not playback; start the filter fresh.
                self.correction.reset_measurements();
                log::debug!(
                    "Clock sync settled; corrections enabled: callback={}, suppressed_during_warmup={}, generation={}",
                    self.stats.callbacks,
                    self.stats.warmup_suppressed_corrections,
                    generation,
                );
            }

            if force_reanchor {
                let mut reanchor_applied = false;
                // Anchor from the delta floor, not this wake's
                // reading (see min_playback_delta_us above). The
                // floor already includes this callback's sample,
                // so it is never u64::MAX here.
                let anchor_instant =
                    callback_instant + Duration::from_micros(self.correction.min_playback_delta_us);
                let handoff_instant = if self.started {
                    anchor_instant
                } else {
                    // Startup handoff: anchor to `+ handoff_delta` (this buffer's
                    // end = the next callback's start) so the next start gate sees
                    // `expected ≈ playback_instant`. Playing now would misalign the
                    // cursor by one buffer, so we stay silent for this one period.
                    let handoff_delta = Duration::from_secs_f64(frames as f64 / sample_rate as f64);
                    anchor_instant + handoff_delta
                };
                let client_micros =
                    sync.instant_to_client_micros(handoff_instant) + delay_us as i64;
                if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                    let mut queue = self.queue.lock();
                    if queue.generation == generation && queue.initialized {
                        if let Some(cursor_us) =
                            queue.first_playable_cursor_at_or_after(server_time)
                        {
                            queue.cursor_us = cursor_us;
                            queue.cursor_remainder = 0;
                            queue.force_reanchor = false;
                            effective_cursor_us = cursor_us;
                            reanchor_applied = true;
                            self.schedule = CorrectionSchedule::default();
                            self.insert_counter = 0;
                            self.drop_counter = 0;
                            self.reader.reset();
                            // The cursor just jumped (e.g. a
                            // static-delay change, which does not
                            // bump the generation); prior
                            // measurements describe the old
                            // timeline.
                            self.correction.reset_measurements();
                            log::debug!(
                                "Sync reanchor applied: cursor reset to server_time={cursor_us}µs"
                            );
                        } else if !self.handoff_warned {
                            self.handoff_warned = true;
                            log::warn!(
                                "Sync reanchor: no playable buffer at or after \
                                 server_time={server_time}µs — staying silent"
                            );
                        }
                    }
                }

                if !reanchor_applied || !self.started {
                    self.stats.silent_callbacks += 1;
                    if trace_logging && should_log_sample(self.stats.silent_callbacks) {
                        log::trace!(
                            "Audio callback silent during reanchor: callback={}, silent_callback={}, reanchor_applied={}, started={}, queued={:.1}ms, buffers={}, generation={}",
                            self.stats.callbacks,
                            self.stats.silent_callbacks,
                            reanchor_applied,
                            self.started,
                            us_to_ms(queued_us),
                            queued_buffers,
                            generation,
                        );
                    }
                    self.output.emit_silence(out);
                    return;
                }
            }

            if let Some(expected_instant) =
                sync.server_to_local_instant_with_latency(effective_cursor_us, delay_us)
            {
                // Pre-start only: hold silence until the cursor's
                // scheduled instant. After start, "early" readings
                // are jitter — injecting silence here caused real
                // dropouts (audible blips); the planner handles
                // sustained earliness instead.
                let early_window = Duration::from_millis(1);
                if !self.started && playback_instant + early_window < expected_instant {
                    self.stats.silent_callbacks += 1;
                    if trace_logging && should_log_sample(self.stats.silent_callbacks) {
                        let early_us = expected_instant
                            .duration_since(playback_instant)
                            .as_micros() as u64;
                        log::trace!(
                            "Audio callback early; emitting silence: callback={}, silent_callback={}, early={:.1}ms, cursor={}µs, queued={:.1}ms, buffers={}, generation={}",
                            self.stats.callbacks,
                            self.stats.silent_callbacks,
                            us_to_ms(early_us),
                            effective_cursor_us,
                            us_to_ms(queued_us),
                            queued_buffers,
                            generation,
                        );
                    }
                    self.output.emit_silence(out);
                    return;
                }
                if !self.started {
                    self.started = true;
                    self.correction.start(callback_instant);
                    log::debug!(
                        "Audio playback started: callback={}, cursor={}µs, queued={:.1}ms, buffers={}, silent_callbacks_before_start={}, sync_lock_misses={}",
                        self.stats.callbacks,
                        effective_cursor_us,
                        us_to_ms(queued_us),
                        queued_buffers,
                        self.stats.silent_callbacks,
                        self.stats.sync_lock_misses,
                    );
                }

                let raw_error_us = if playback_instant >= expected_instant {
                    playback_instant
                        .duration_since(expected_instant)
                        .as_micros() as i64
                } else {
                    -(expected_instant
                        .duration_since(playback_instant)
                        .as_micros() as i64)
                };
                // Callback delay noise can sit above gapless alignment.
                // Plan against the error window's floor, not one snapshot.
                let decision = self.correction.plan(
                    callback_instant,
                    raw_error_us,
                    sample_rate,
                    self.schedule.is_correcting(),
                    sync_settled,
                );
                let error_us = decision.error_us;
                let ungated_schedule = decision.ungated;
                let gated_schedule = decision.gated;
                if gated_schedule != ungated_schedule {
                    self.stats.gate_suppressed_corrections += 1;
                    if trace_logging && should_log_sample(self.stats.gate_suppressed_corrections) {
                        log::trace!(
                            "Sync correction awaiting sustained error: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, generation={}",
                            self.stats.callbacks,
                            self.stats.gate_suppressed_corrections,
                            error_us as f64 / 1000.0,
                            raw_error_us as f64 / 1000.0,
                            generation,
                        );
                    }
                }
                // Settling output delay and a converging clock can mimic drift.
                // Log the same suppression policy that selected the schedule.
                if decision.output_delay_transient {
                    if gated_schedule.is_correcting() {
                        self.stats.delay_transient_suppressed_corrections += 1;
                        if trace_logging
                            && should_log_sample(self.stats.delay_transient_suppressed_corrections)
                        {
                            log::trace!(
                                "Sync correction suppressed while output delay settles: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, playback_delta={:.1}ms, generation={}",
                                self.stats.callbacks,
                                self.stats.delay_transient_suppressed_corrections,
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                us_to_ms(playback_delta_us),
                                generation,
                            );
                        }
                    }
                } else if decision.clock_warmup_suppressed && gated_schedule.is_correcting() {
                    self.stats.warmup_suppressed_corrections += 1;
                    if trace_logging && should_log_sample(self.stats.warmup_suppressed_corrections)
                    {
                        log::trace!(
                            "Sync correction suppressed during clock warm-up: callback={}, suppressed={}, error={:.3}ms, raw_error={:.3}ms, generation={}",
                            self.stats.callbacks,
                            self.stats.warmup_suppressed_corrections,
                            error_us as f64 / 1000.0,
                            raw_error_us as f64 / 1000.0,
                            generation,
                        );
                    }
                }
                let new_schedule = decision.schedule;
                if new_schedule != self.schedule {
                    if new_schedule.is_correcting() != self.schedule.is_correcting() {
                        if new_schedule.is_correcting() {
                            self.stats.correction_engagements += 1;
                            log::debug!(
                                "Sync correction engaged: error={:.3}ms, raw_error={:.3}ms, insert_every={}, drop_every={}, reanchor={}, callback={}, generation={}",
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                new_schedule.insert_every_n_frames,
                                new_schedule.drop_every_n_frames,
                                new_schedule.reanchor,
                                self.stats.callbacks,
                                generation,
                            );
                        } else {
                            // The floor lags rises, so error= may
                            // read worse than raw_error= here;
                            // expected, not a bug.
                            log::debug!(
                                "Sync correction disengaged: error={:.3}ms, raw_error={:.3}ms, callback={}, generation={}",
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                self.stats.callbacks,
                                generation,
                            );
                        }
                    }
                    if new_schedule.is_correcting() {
                        // The cadence is re-planned as the error
                        // converges, which can change the schedule
                        // on every callback. Sample the updates so
                        // each correction episode logs its first
                        // few adjustments and then a heartbeat;
                        // engage/disengage transitions are logged
                        // at debug above and reanchor execution is
                        // logged where it is applied below.
                        self.stats.correction_updates += 1;
                        if trace_logging && should_log_sample(self.stats.correction_updates) {
                            log::trace!(
                                "Sync correction updated: callback={}, correction_update={}, error={:.3}ms, raw_error={:.3}ms, insert_every={}, drop_every={}, reanchor={}, queued={:.1}ms, generation={}",
                                self.stats.callbacks,
                                self.stats.correction_updates,
                                error_us as f64 / 1000.0,
                                raw_error_us as f64 / 1000.0,
                                new_schedule.insert_every_n_frames,
                                new_schedule.drop_every_n_frames,
                                new_schedule.reanchor,
                                us_to_ms(queued_us),
                                generation,
                            );
                        }
                    } else {
                        self.stats.correction_updates = 0;
                    }
                    self.schedule = new_schedule;
                    self.insert_counter = self.schedule.insert_every_n_frames;
                    self.drop_counter = self.schedule.drop_every_n_frames;
                }

                if self.schedule.reanchor {
                    // Mirror of the start-gate subtraction: audio
                    // emitted now is heard `delay_us` later, so
                    // anchor the cursor to that hear-instant —
                    // derived from the delta floor, not this
                    // wake's reading (see min_playback_delta_us).
                    let anchor_instant = callback_instant
                        + Duration::from_micros(self.correction.min_playback_delta_us);
                    let client_micros =
                        sync.instant_to_client_micros(anchor_instant) + delay_us as i64;
                    if let Some(server_time) = sync.client_to_server_micros(client_micros) {
                        let mut queue = self.queue.lock();
                        queue.cursor_us = server_time;
                        queue.cursor_remainder = 0;
                        log::debug!(
                            "Sync reanchor applied: cursor reset to server_time={server_time}µs"
                        );
                    }
                    self.schedule = CorrectionSchedule::default();
                    self.insert_counter = 0;
                    self.drop_counter = 0;
                    self.reader.reset();
                    self.stats.correction_updates = 0;
                    // The cursor just jumped; prior measurements
                    // describe the old timeline.
                    self.correction.reset_measurements();
                }
            } else if self.schedule.is_correcting() {
                // Conversions went dark (the implausible-drift
                // safety net): stop correcting rather than
                // resample blind on the stale cadence.
                log::debug!(
                    "Sync conversions unavailable; clearing correction schedule: callback={}, generation={}",
                    self.stats.callbacks,
                    generation,
                );
                self.schedule = CorrectionSchedule::default();
                self.insert_counter = 0;
                self.drop_counter = 0;
                self.stats.correction_updates = 0;
                self.correction.reset_measurements();
            }
        }

        // If playback hasn't started yet (clock sync not converged,
        // lock contention, or pre-start gate active), output silence.
        // Audio data stays in the ring buffer for when sync converges
        // and reanchor positions the cursor correctly.
        if !self.started {
            self.stats.silent_callbacks += 1;
            if trace_logging && should_log_sample(self.stats.silent_callbacks) {
                log::trace!(
                    "Audio callback silent before start: callback={}, silent_callback={}, cursor_present={}, queued={:.1}ms, buffers={}, generation={}",
                    self.stats.callbacks,
                    self.stats.silent_callbacks,
                    cursor_us.is_some(),
                    us_to_ms(queued_us),
                    queued_buffers,
                    generation,
                );
            }
            self.output.emit_silence(out);
            return;
        }

        let (callback_underrun_frames, queued_after_us, buffers_after) = {
            let mut queue = self.queue.lock();
            let mut missing_frames = 0u64;
            let mut out_index = 0;

            let smooth = self.correction_mode == CorrectionMode::Smooth;
            let step = read_step(&self.schedule);

            for _ in 0..frames {
                if smooth {
                    let frame_out = &mut out[out_index..out_index + channels];
                    out_index += channels;
                    if !self.reader.read_frame(
                        step,
                        &mut self.last_frame,
                        &mut queue,
                        channels,
                        sample_rate,
                        frame_out,
                    ) {
                        missing_frames += 1;
                    }
                    continue;
                }

                if self.schedule.drop_every_n_frames > 0 {
                    self.drop_counter = self.drop_counter.saturating_sub(1);
                    if self.drop_counter == 0 {
                        // Discard one frame to catch up
                        let _ = queue.next_frame(channels, sample_rate);
                        self.drop_counter = self.schedule.drop_every_n_frames;
                        // Get and output the next frame (don't repeat last_frame)
                        if let Some(frame) = queue.next_frame(channels, sample_rate) {
                            self.last_frame.copy_from_slice(frame);
                            for sample in frame {
                                out[out_index] = f32::from_sample(*sample);
                                out_index += 1;
                            }
                        } else {
                            for sample in &self.last_frame {
                                out[out_index] = f32::from_sample(*sample);
                                out_index += 1;
                            }
                        }
                        continue;
                    }
                }

                if self.schedule.insert_every_n_frames > 0 {
                    self.insert_counter = self.insert_counter.saturating_sub(1);
                    if self.insert_counter == 0 {
                        self.insert_counter = self.schedule.insert_every_n_frames;
                        for sample in &self.last_frame {
                            out[out_index] = f32::from_sample(*sample);
                            out_index += 1;
                        }
                        continue;
                    }
                }

                if let Some(frame) = queue.next_frame(channels, sample_rate) {
                    self.last_frame.copy_from_slice(frame);
                    for sample in frame {
                        out[out_index] = f32::from_sample(*sample);
                        out_index += 1;
                    }
                } else {
                    missing_frames += 1;
                    for _ in 0..channels {
                        out[out_index] = 0.0;
                        out_index += 1;
                    }
                }
            }

            let (queued_after_us, buffers_after) = if debug_logging {
                (
                    queue.queued_duration_us(channels, sample_rate),
                    queue.buffer_count(),
                )
            } else {
                (0, 0)
            };
            (missing_frames, queued_after_us, buffers_after)
        }; // queue lock dropped before user callback

        let recovered =
            callback_underrun_frames == 0 && self.stats.consecutive_underrun_callbacks > 0;
        if callback_underrun_frames > 0 {
            self.stats.underrun_frames += callback_underrun_frames;
            self.stats.underrun_callbacks += 1;
            self.stats.consecutive_underrun_callbacks += 1;

            // Per-generation totals reset on stream changes, so
            // every stream logs its first few underruns at debug.
            // That is intentional: startup underruns after a
            // clear/track change are the main diagnostic.
            if debug_logging
                && (should_log_sample(self.stats.underrun_callbacks)
                    || should_log_sample(self.stats.consecutive_underrun_callbacks))
            {
                log::debug!(
                    "Audio underrun: callback={}, missing_frames={} ({:.1}ms), queued_before={:.1}ms, queued_after={:.1}ms, buffers_after={}, cursor={:?}µs, generation={}, underrun_frames={}, underrun_callbacks={}, consecutive_underrun_callbacks={}",
                    self.stats.callbacks,
                    callback_underrun_frames,
                    callback_underrun_frames as f64 * 1000.0 / sample_rate as f64,
                    us_to_ms(queued_us),
                    us_to_ms(queued_after_us),
                    buffers_after,
                    cursor_us,
                    generation,
                    self.stats.underrun_frames,
                    self.stats.underrun_callbacks,
                    self.stats.consecutive_underrun_callbacks,
                );
            }
        } else if recovered {
            let underrun_run = self.stats.consecutive_underrun_callbacks;
            self.stats.consecutive_underrun_callbacks = 0;
            log::debug!(
                "Audio underrun recovered: callback={}, consecutive_underrun_callbacks={}, underrun_frames={}, queued_after={:.1}ms, buffers_after={}, generation={}",
                self.stats.callbacks,
                underrun_run,
                self.stats.underrun_frames,
                us_to_ms(queued_after_us),
                buffers_after,
                generation,
            );
        }

        // Edge-triggered low-queue warnings with hysteresis, so a
        // queue hovering at one boundary cannot flood the log.
        if debug_logging {
            if !self.stats.queue_low && queued_after_us < QUEUE_LOW_WATER_US {
                self.stats.queue_low = true;
                log::debug!(
                    "Playback queue low: queued={:.1}ms, buffers={}, callback={}, underrun_frames={}, generation={}",
                    us_to_ms(queued_after_us),
                    buffers_after,
                    self.stats.callbacks,
                    self.stats.underrun_frames,
                    generation,
                );
            } else if self.stats.queue_low && queued_after_us >= QUEUE_RECOVERED_WATER_US {
                self.stats.queue_low = false;
                log::debug!(
                    "Playback queue recovered: queued={:.1}ms, buffers={}, callback={}, generation={}",
                    us_to_ms(queued_after_us),
                    buffers_after,
                    self.stats.callbacks,
                    generation,
                );
            }
        }

        self.output.emit(out);

        // One sampled health line per rendered callback, emitted
        // after gain and the user process callback so it describes
        // the audio actually delivered.
        if trace_logging
            && callback_underrun_frames == 0
            && !recovered
            && should_log_sample(self.stats.callbacks)
        {
            let peak_abs = out.iter().map(|sample| sample.abs()).fold(0.0, f32::max);
            log::trace!(
                "Audio callback rendered: callback={}, frames={}, queued_before={:.1}ms, queued_after={:.1}ms, buffers_after={}, peak_abs={:.6}, generation={}",
                self.stats.callbacks,
                frames,
                us_to_ms(queued_us),
                us_to_ms(queued_after_us),
                buffers_after,
                peak_abs,
                generation,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        read_step, static_delay_ms_to_us, FractionalReader, PlaybackQueue, MAX_STATIC_DELAY_MS,
    };
    use crate::audio::sync_correction::CorrectionSchedule;
    use crate::audio::{AudioBuffer, AudioFormat, Codec};
    use dasp_sample::Sample;
    use std::sync::Arc;

    /// Standard test format: 48kHz stereo 24-bit PCM.
    fn test_format() -> AudioFormat {
        AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        }
    }

    /// Mono variant of [`test_format`].
    fn test_format_mono() -> AudioFormat {
        AudioFormat {
            channels: 1,
            ..test_format()
        }
    }

    #[test]
    fn test_static_delay_ms_to_us_converts_and_clamps() {
        // Nominal values convert milliseconds to microseconds.
        assert_eq!(static_delay_ms_to_us(0), 0);
        assert_eq!(static_delay_ms_to_us(100), 100_000);
        assert_eq!(
            static_delay_ms_to_us(MAX_STATIC_DELAY_MS),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );

        // Out-of-range values clamp to the maximum rather than overflow or wrap.
        assert_eq!(
            static_delay_ms_to_us(MAX_STATIC_DELAY_MS + 1),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );
        assert_eq!(
            static_delay_ms_to_us(u16::MAX),
            MAX_STATIC_DELAY_MS as u64 * 1_000
        );
    }

    #[test]
    fn test_queue_clear_bumps_generation() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 96];
        queue.push(AudioBuffer {
            timestamp: 1234,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let before = queue.generation;
        queue.clear();
        assert_ne!(queue.generation, before);
        assert!(queue.queue.is_empty());
        assert!(!queue.initialized);
    }

    #[test]
    fn test_queue_drops_stale_buffers() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Use distinct sample values so we can verify which buffer was returned.
        // 4800 stereo frames at 48kHz = 100ms per buffer.
        // With cursor at 150ms, the first buffer (ts=0, ends at 100ms) is stale.
        let stale_samples: Vec<i32> = (0..4800 * 2).map(|_| 111).collect();
        let fresh_samples: Vec<i32> = (0..4800 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(stale_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 200_000,
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        queue.cursor_us = 150_000;
        queue.initialized = true;

        // Copy the frame data so we can release the mutable borrow on queue.
        let frame_data: Vec<i32> = queue
            .next_frame(2, 48_000)
            .expect("expected a frame")
            .to_vec();
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);
        // Verify we got the fresh buffer's data, not the stale one
        assert_eq!(frame_data[0], 222);
        assert_eq!(frame_data[1], 222);
    }

    #[test]
    fn test_queue_push_sorts_by_timestamp() {
        let mut queue = PlaybackQueue::new();
        let format = test_format_mono();

        // Push out of order: 300, 100, 200
        for ts in [300_000i64, 100_000, 200_000] {
            let samples = vec![ts as i32; 48]; // 1ms of mono
            queue.push(AudioBuffer {
                timestamp: ts,
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
        }

        // Reset cursor so stale-buffer-dropping doesn't interfere with
        // the sort-order verification (in real usage, sync reanchor sets
        // cursor before playback begins).
        queue.cursor_us = 0;

        // Drain and verify sorted order
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 100_000);

        // Exhaust the first buffer (48 frames)
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the second buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 200_000);

        // Exhaust the second buffer
        for _ in 1..48 {
            queue.next_frame(1, 48_000);
        }
        // Next frame should come from the third buffer
        let _ = queue.next_frame(1, 48_000);
        assert_eq!(queue.current.as_ref().unwrap().timestamp, 300_000);
    }

    #[test]
    fn test_queue_cursor_advances_correctly() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 480 stereo frames = 10ms at 48kHz
        let num_frames = 480;
        let samples = vec![i32::EQUILIBRIUM; num_frames * 2];
        let start_ts = 1_000_000i64; // 1 second
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Consume all frames
        for _ in 0..num_frames {
            let _ = queue.next_frame(2, 48_000);
        }

        // 480 frames at 48kHz = 10,000us = 10ms
        let expected_end = start_ts + 10_000;
        assert_eq!(
            queue.cursor_us,
            expected_end,
            "cursor should advance by exactly 10ms (10000us), got delta={}",
            queue.cursor_us - start_ts
        );
    }

    /// Ramp step per input frame, in sample units.
    const RAMP_UNIT: i32 = 1_000;

    /// A queue holding one stereo ramp of `frames` frames from `start_ts`:
    /// frame `i` is `i * RAMP_UNIT` on the left and its negation on the right.
    fn ramp_queue(start_ts: i64, frames: usize) -> PlaybackQueue {
        let mut queue = PlaybackQueue::new();
        let samples: Vec<i32> = (0..frames as i32)
            .flat_map(|i| [i * RAMP_UNIT, -i * RAMP_UNIT])
            .collect();
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format: test_format(),
        });
        queue
    }

    /// A rendered sample as a position on the ramp, in input frames.
    fn ramp_position(sample: f32) -> f64 {
        f64::from(sample) * 2_147_483_648.0 / f64::from(RAMP_UNIT)
    }

    #[test]
    fn test_fractional_reader_resamples_a_ramp_without_repeats_or_skips() {
        let schedules = [
            CorrectionSchedule {
                drop_every_n_frames: 200,
                ..CorrectionSchedule::default()
            },
            CorrectionSchedule {
                insert_every_n_frames: 200,
                ..CorrectionSchedule::default()
            },
            CorrectionSchedule {
                drop_every_n_frames: 25,
                ..CorrectionSchedule::default()
            },
        ];
        const PLAIN: usize = 10;
        const CORRECTING: usize = 96_000; // 2s at 48kHz
        const AFTER: usize = 4_800;
        let start_ts = 1_000_000i64;
        let total_frames = 110_000;

        for schedule in schedules {
            let step = read_step(&schedule);
            let mut queue = ramp_queue(start_ts, total_frames);
            let mut reader = FractionalReader::new(2);
            let mut last_frame = vec![i32::EQUILIBRIUM; 2];
            let mut out = [0.0f32; 2];
            let mut read = |step: f64, queue: &mut PlaybackQueue| {
                assert!(reader.read_frame(step, &mut last_frame, queue, 2, 48_000, &mut out));
                assert!((out[0] + out[1]).abs() < 1e-6, "channels stay apart");
                ramp_position(out[0])
            };

            // Frame by frame first: whole input frames.
            let mut positions = Vec::new();
            for _ in 0..PLAIN {
                positions.push(read(1.0, &mut queue));
            }
            assert_eq!(positions, (0..PLAIN).map(|i| i as f64).collect::<Vec<_>>());

            // Correcting: each output moves exactly `step` along the ramp, so
            // no input sample is repeated or skipped.
            let mut previous = positions[PLAIN - 1];
            for i in 0..CORRECTING {
                let position = read(step, &mut queue);
                assert!(
                    (position - previous - step).abs() < 0.05,
                    "step {step}: output {i} moved {} frames",
                    position - previous
                );
                previous = position;
            }

            // Input is consumed at the planned rate, give or take the
            // lookahead frame, and the cursor accounts for every frame read.
            let planned = PLAIN as f64 + CORRECTING as f64 * step;
            let consumed = total_frames - queue.queued_frames(2);
            assert!(
                (consumed as f64 - planned).abs() <= 1.0,
                "step {step}: consumed {consumed} frames, planned {planned}"
            );
            assert_eq!(
                queue.cursor_us,
                start_ts + (consumed as i64 * 1_000_000) / 48_000,
                "one frame read advances the cursor one frame"
            );

            // Back to frame by frame: the correction finishes on whole
            // frames, never moving further from one frame per output than
            // its own step did, then whole frames follow one at a time.
            let excess = (step - 1.0).abs();
            let finish_frames = (1.0 / excess).ceil() as usize + 1;
            for i in 0..AFTER {
                let position = read(1.0, &mut queue);
                let moved = position - previous;
                assert!(
                    (moved - 1.0).abs() <= 2.0 * excess + 0.05,
                    "step {step}: output {i} after the correction moved {moved} frames"
                );
                if i >= finish_frames {
                    assert!((moved - 1.0).abs() < 0.05, "output {i} moved {moved}");
                    assert!((position - position.round()).abs() < 0.05);
                }
                previous = position;
            }
        }
    }

    #[test]
    fn test_fractional_reader_underrun_is_silent_and_resumes() {
        let mut queue = ramp_queue(1_000_000, 3);
        let mut reader = FractionalReader::new(2);
        let mut last_frame = vec![i32::EQUILIBRIUM; 2];
        let mut out = [f32::NAN; 2];
        let step = 1.5;
        // Reading starts from the frame before the queue's first.
        assert!(reader.read_frame(step, &mut last_frame, &mut queue, 2, 48_000, &mut out));
        assert_eq!(ramp_position(out[0]), 0.5);
        assert!(reader.read_frame(step, &mut last_frame, &mut queue, 2, 48_000, &mut out));
        assert_eq!(ramp_position(out[0]), 2.0);
        // All three frames are read; the next position needs a fourth.
        assert!(!reader.read_frame(step, &mut last_frame, &mut queue, 2, 48_000, &mut out));
        assert_eq!(out, [0.0, 0.0]);
        assert_eq!(queue.cursor_us, 1_000_000 + 3 * 1_000_000 / 48_000);

        // Audio returns: reading resumes after the last frame read.
        queue.push(AudioBuffer {
            timestamp: queue.cursor_us,
            samples: Arc::from(vec![7 * RAMP_UNIT, -7 * RAMP_UNIT].into_boxed_slice()),
            format: test_format(),
        });
        assert!(reader.read_frame(1.0, &mut last_frame, &mut queue, 2, 48_000, &mut out));
        assert_eq!(ramp_position(out[0]), 7.0);
    }

    #[test]
    fn test_cursor_does_not_advance_during_underrun() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Push one buffer to initialize the cursor
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo
        let start_ts = 1_000_000i64;
        queue.push(AudioBuffer {
            timestamp: start_ts,
            samples: Arc::from(samples.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all frames
        for _ in 0..480 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        let cursor_after_drain = queue.cursor_us;

        // Queue is now empty. Calling next_frame should return None
        // and NOT advance the cursor.
        for _ in 0..1000 {
            assert!(queue.next_frame(2, 48_000).is_none());
        }
        assert_eq!(
            queue.cursor_us,
            cursor_after_drain,
            "cursor must not advance during underrun; advanced by {}us",
            queue.cursor_us - cursor_after_drain
        );

        // Push a new buffer after the underrun. It should NOT be
        // dropped as stale — the cursor hasn't raced ahead.
        let fresh_samples: Vec<i32> = (0..480 * 2).map(|_| 999).collect();
        queue.push(AudioBuffer {
            timestamp: cursor_after_drain, // starts right where we left off
            samples: Arc::from(fresh_samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("buffer should not be dropped as stale");
        assert_eq!(frame[0], 999, "should get the fresh buffer, not stale data");
    }

    #[test]
    fn test_push_initializes_cursor_from_first_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        assert!(!queue.initialized);
        assert_eq!(queue.cursor_us, 0);

        let samples = vec![i32::EQUILIBRIUM; 96];
        queue.push(AudioBuffer {
            timestamp: 500_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert!(queue.initialized);
        assert_eq!(queue.cursor_us, 500_000);
        assert_eq!(queue.cursor_remainder, 0);
    }

    #[test]
    fn test_push_does_not_regress_cursor_after_init() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 96];

        // First buffer at 500ms — initializes cursor
        queue.push(AudioBuffer {
            timestamp: 500_000,
            samples: Arc::from(samples.clone().into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.cursor_us, 500_000);

        // Consume a frame so cursor advances past init
        let _ = queue.next_frame(2, 48_000);
        let cursor_after_consume = queue.cursor_us;
        assert!(cursor_after_consume > 500_000);

        // Push an earlier buffer — cursor must NOT regress
        queue.push(AudioBuffer {
            timestamp: 200_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });
        assert_eq!(
            queue.cursor_us, cursor_after_consume,
            "cursor must not regress after playback has started"
        );
    }

    #[test]
    fn test_first_playable_cursor_skips_stale_audio() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_first_playable_cursor_waits_for_future_audio() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 20_000,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(20_000));
    }

    #[test]
    fn test_first_playable_cursor_uses_current_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        // Pull one frame so the buffer moves from `queue` into `current`,
        // exercising the `current` branch of first_playable_cursor_at_or_after.
        let _ = queue.next_frame(2, 48_000);
        assert!(queue.current.is_some());
        assert!(queue.queue.is_empty());

        assert_eq!(queue.first_playable_cursor_at_or_after(5_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_first_playable_cursor_does_not_rewind_current_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();
        let samples = vec![i32::EQUILIBRIUM; 480 * 2]; // 10ms stereo

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        for _ in 0..240 {
            let _ = queue.next_frame(2, 48_000);
        }
        assert_eq!(queue.cursor_us, 5_000);

        assert_eq!(queue.first_playable_cursor_at_or_after(1_000), Some(5_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(6_000), Some(6_000));
        assert_eq!(queue.first_playable_cursor_at_or_after(10_000), None);
    }

    #[test]
    fn test_next_frame_skips_into_overlapping_buffer() {
        // Simulates a backward timestamp jump from a server timeline rebase.
        // Buffer A is 50ms (2400 frames stereo at 48kHz). After consuming A
        // the cursor is at 50ms. Buffer B arrives at 25ms — the skip logic
        // should jump 25ms (1200 frames) into B.
        //
        // B is pushed AFTER A is consumed so dedup doesn't apply (A is no
        // longer in the queue).
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        let buf_a: Vec<i32> = (0..2400 * 2).map(|_| 111).collect();
        let buf_b: Vec<i32> = (0..2400 * 2)
            .map(|i| {
                // First half (1200 frames) = 222, second half = 333
                if i < 2400 {
                    222
                } else {
                    333
                }
            })
            .collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(buf_a.into_boxed_slice()),
            format: format.clone(),
        });

        // Consume all of buffer A (2400 frames). Cursor advances to 50000µs.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }
        assert_eq!(queue.cursor_us, 50_000);

        // Push B after A is consumed — no dedup, tests skip logic only.
        // push() no longer regresses cursor_us after init, so cursor stays
        // at 50000 and the skip logic activates naturally.
        queue.push(AudioBuffer {
            timestamp: 25_000,
            samples: Arc::from(buf_b.into_boxed_slice()),
            format,
        });

        // Buffer B starts at 25ms but cursor is at 50ms, so 25ms (1200 frames)
        // should be skipped. First returned frame should be Sample(333).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get a frame from buffer B");
        assert_eq!(
            frame[0], 333,
            "expected skip into second half of buffer B (past the overlap), \
             got first half — backward-timestamped audio was replayed"
        );
    }

    #[test]
    fn test_next_frame_no_skip_when_buffer_starts_at_or_after_cursor() {
        // Verify that the skip logic doesn't activate for normal (non-overlapping)
        // buffers — only for buffers that start before the cursor.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 2400 frames = 50ms per buffer. Adjacent, non-overlapping.
        let samples_a: Vec<i32> = (0..2400 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..2400 * 2).map(|_| 222).collect();

        // Two consecutive, non-overlapping buffers.
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 50_000, // starts exactly where A ends
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Consume all of buffer A.
        for _ in 0..2400 {
            assert!(queue.next_frame(2, 48_000).is_some());
        }

        // Buffer B starts at cursor (50000µs) — no skip should occur.
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should get first frame of buffer B");
        assert_eq!(
            frame[0], 222,
            "buffer B should play from the start (no skip needed)"
        );
    }

    #[test]
    fn test_push_dedup_replaces_overlapping_buffer() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Buffer A: 10ms at ts=0 (480 stereo frames)
        let samples_a: Vec<i32> = (0..480 * 2).map(|_| 111).collect();
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 1);

        // Buffer B: 10ms at ts=5000 (5ms) — overlaps A's range [0, 10000)
        let samples_b: Vec<i32> = (0..480 * 2).map(|_| 222).collect();
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Should replace A, not add a second entry
        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].samples[0], 222);
    }

    #[test]
    fn test_push_dedup_no_false_positive_small_chunks() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Two adjacent 5ms chunks (240 stereo frames each).
        // Chunk A: [0, 5000), Chunk B: [5000, 10000) — no overlap.
        let samples_a: Vec<i32> = (0..240 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..240 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        // Both should be kept — they're adjacent, not overlapping
        assert_eq!(queue.queue.len(), 2);
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_push_keeps_chunks_on_44_1k_floor_timestamp_grid() {
        // Regression for the field "continuous popping" report: aiosendspin's
        // 25ms chunks at 44.1kHz are 1102 frames = 24988.66µs, and its
        // floor-based timestamp grid advances 24988µs for a third of chunks
        // while `duration_us` rounds every chunk to 24989µs. Those chunks
        // start 1µs "inside" their predecessor; the phantom overlap must not
        // evict anything (pre-fix it discarded ~34% of all queued audio).
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };

        // Mirror the server's residue arithmetic:
        // delta, residue = divmod(residue + frames * 1e6, rate).
        let mut ts = 282_697_405_880_i64; // first chunk ts from the field trace
        let mut residue = 0_i64;
        for _ in 0..12 {
            let samples: Vec<i32> = vec![0; 1102 * 2];
            queue.push(AudioBuffer {
                timestamp: ts,
                samples: Arc::from(samples.into_boxed_slice()),
                format: format.clone(),
            });
            residue += 1102 * 1_000_000;
            ts += residue / 44_100;
            residue %= 44_100;
        }

        assert_eq!(
            queue.queue.len(),
            12,
            "sub-frame timestamp jitter must never evict queued audio"
        );
    }

    #[test]
    fn test_push_keeps_floor_grid_chunks_at_all_supported_rates() {
        // The 44.1k regression swept across the whole rate family. 25ms
        // chunks are floor(rate/40) frames; the server grid advances by
        // floor-based (divmod) deltas. The mismatch between that grid and
        // our round-to-nearest duration_us is at most 1µs, which must stay
        // below the one-frame eviction threshold at every rate. (Only 44.1k
        // and 22.05k actually exhibit the mismatch; the rest divide 25ms
        // evenly and are exact.)
        for &rate in &[
            8_000_u32, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000,
            176_400, 192_000,
        ] {
            let mut queue = PlaybackQueue::new();
            let format = AudioFormat {
                codec: Codec::Pcm,
                sample_rate: rate,
                channels: 2,
                bit_depth: 16,
                codec_header: None,
            };
            let frames = (rate / 40) as usize;
            let mut ts = 1_000_000_i64;
            let mut residue = 0_i64;
            for _ in 0..16 {
                queue.push(AudioBuffer {
                    timestamp: ts,
                    samples: Arc::from(vec![0_i32; frames * 2].into_boxed_slice()),
                    format: format.clone(),
                });
                residue += frames as i64 * 1_000_000;
                ts += residue / i64::from(rate);
                residue %= i64::from(rate);
            }
            assert_eq!(
                queue.queue.len(),
                16,
                "rate {rate}: floor-grid timestamp jitter must not evict chunks"
            );
        }
    }

    #[test]
    fn test_push_dedup_still_evicts_one_frame_overlap_at_44_1k() {
        // Counterpart to the floor-grid regression: an overlap of a full
        // frame or more is real duplicate audio and must still dedup.
        let mut queue = PlaybackQueue::new();
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 44_100,
            channels: 2,
            bit_depth: 16,
            codec_header: None,
        };

        let samples_a: Vec<i32> = vec![111; 1102 * 2];
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        // Chunk A spans [0, 24989). A chunk rebased 1ms back into it
        // overlaps by ~1ms ≫ one frame (23µs): evict A.
        let samples_b: Vec<i32> = vec![222; 1102 * 2];
        queue.push(AudioBuffer {
            timestamp: 23_989,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 23_989);
    }

    #[test]
    fn test_push_dedup_removes_all_overlapping() {
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Buffer A: 10ms at ts=0 — range [0, 10000)
        let samples_a: Vec<i32> = (0..480 * 2).map(|_| 111).collect();
        // Buffer B: 10ms at ts=12000 — range [12000, 22000). No overlap with A.
        let samples_b: Vec<i32> = (0..480 * 2).map(|_| 222).collect();

        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format: format.clone(),
        });
        queue.push(AudioBuffer {
            timestamp: 12_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        assert_eq!(queue.queue.len(), 2);

        // Buffer C: 20ms at ts=9000 — range [9000, 29000).
        // Overlaps both A (9000 < 10000 && 0 < 29000) and B (9000 < 22000 && 12000 < 29000).
        // Both stale buffers should be removed — the server will send fresh
        // data for any gaps. Keeping either would cause duplicate audio.
        let samples_c: Vec<i32> = (0..960 * 2).map(|_| 333).collect();
        queue.push(AudioBuffer {
            timestamp: 9_000,
            samples: Arc::from(samples_c.into_boxed_slice()),
            format,
        });

        assert_eq!(queue.queue.len(), 1);
        assert_eq!(queue.queue[0].timestamp, 9_000);
        assert_eq!(queue.queue[0].samples[0], 333);
    }

    #[test]
    fn test_push_keeps_adjacent_buffers_inserted_in_reverse_order() {
        // When buffer B is pushed *after* buffer A and they abut at the
        // boundary (B.ts == A.end), the existing dedup check keeps both
        // because `b.timestamp < new_end` is false at the boundary. The
        // symmetric case — pushing the *earlier* buffer second — must also
        // keep both: when we push the earlier buffer A and retain() walks
        // B, we see `B.ts == new_end` (A.end). The dedup has to treat the
        // boundary as *not* overlapping, otherwise we'd evict a buffer that
        // simply abuts — a common pattern when chunks arrive out-of-order.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Two 5ms adjacent chunks: A at [0, 5000), B at [5000, 10000).
        let samples_a: Vec<i32> = (0..240 * 2).map(|_| 111).collect();
        let samples_b: Vec<i32> = (0..240 * 2).map(|_| 222).collect();

        // Push the *later* buffer (B) first …
        queue.push(AudioBuffer {
            timestamp: 5_000,
            samples: Arc::from(samples_b.into_boxed_slice()),
            format: format.clone(),
        });
        // … then push the *earlier* buffer (A). A.end == B.ts (boundary case).
        queue.push(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples_a.into_boxed_slice()),
            format,
        });

        assert_eq!(
            queue.queue.len(),
            2,
            "adjacent buffers pushed in reverse order should both be kept"
        );
        // Should also be sorted: A first, then B.
        assert_eq!(queue.queue[0].timestamp, 0);
        assert_eq!(queue.queue[1].timestamp, 5_000);
    }

    #[test]
    fn test_next_frame_returns_final_frame_when_skip_lands_at_last_frame() {
        // When the cursor skip lands `self.index` at exactly
        // `samples.len() - channels`, there is still one playable frame at
        // the tail of the buffer. The "is this buffer exhausted?" check in
        // the outer match is `self.index + channels > c.samples.len()` —
        // strictly greater — so `index == samples.len() - channels` falls
        // through to the frame-return path. A non-strict comparison here
        // would silently drop the last frame of every buffer whose skip
        // landed on the final-frame boundary.
        //
        // Setup: 48-frame stereo buffer at ts=0, cursor at 980 µs.
        //   skip_us     = 980 - 0                          = 980
        //   skip_frames = 980 * 48_000 / 1_000_000         = 47
        //   self.index  = 47 * 2                           = 94
        //   samples.len = 48 * 2                           = 96
        //   94 + 2 == 96 — keep buffer, return last frame.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // Distinctive sample values so we can assert *which* frame was returned.
        let samples: Vec<i32> = (0..48 * 2).map(|_| 111).collect();

        queue.initialized = true;
        queue.cursor_us = 980;
        queue.queue.push_back(AudioBuffer {
            timestamp: 0,
            samples: Arc::from(samples.into_boxed_slice()),
            format,
        });

        let frame = queue
            .next_frame(2, 48_000)
            .expect("last frame should be returned, not discarded");
        // Final stereo frame — indices 94 and 95 of the flat sample array.
        assert_eq!(frame, &[111, 111]);
    }

    #[test]
    fn test_skip_past_entire_buffer_does_not_panic() {
        // When the cursor is far ahead of a buffer, the skip logic can set
        // self.index past the buffer's sample count. next_frame must not
        // panic; it should discard the buffer and return the next one.
        let mut queue = PlaybackQueue::new();
        let format = test_format();

        // 1ms buffer (48 stereo frames) at ts=49000. Duration = 1000µs,
        // so it ends at 50000 which is NOT < cursor (50000), surviving
        // the stale-drop. But the skip logic sees ts=49000 < cursor=50000
        // and tries to skip 1ms (48 frames) — exactly the buffer length.
        let short_samples: Vec<i32> = (0..48 * 2).map(|_| 111).collect();
        // Buffer that starts at cursor: 10ms at ts=50000
        let ahead_samples: Vec<i32> = (0..480 * 2).map(|_| 222).collect();

        queue.initialized = true;
        queue.cursor_us = 50_000;
        queue.queue.push_back(AudioBuffer {
            timestamp: 49_000,
            samples: Arc::from(short_samples.into_boxed_slice()),
            format: format.clone(),
        });
        queue.queue.push_back(AudioBuffer {
            timestamp: 50_000,
            samples: Arc::from(ahead_samples.into_boxed_slice()),
            format,
        });

        // The skip tries to skip 1ms (48 frames) into a 48-frame buffer —
        // index lands at the end. Must not panic; should discard the short
        // buffer and return from the next one (ts=50000).
        let frame = queue
            .next_frame(2, 48_000)
            .expect("should return a frame from the next buffer, not panic");
        assert_eq!(frame[0], 222, "expected frame from the ahead buffer");
    }
}

#[cfg(test)]
mod renderer_tests {
    // Drives a renderer the way an output's write loop would, with no
    // device: each render reports how long until its first frame is heard.

    use super::{CorrectionMode, SyncedRenderer, SyncedRendererConfig, SyncedSource};
    use crate::audio::{AudioBuffer, AudioFormat, Codec};
    use crate::sync::{Clock, ClockSync, DefaultClock};
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    const SAMPLE_RATE: u32 = 48_000;
    /// 10ms at [`SAMPLE_RATE`].
    const FRAMES: usize = 480;
    const FRAME_US: i64 = 1_000_000 / SAMPLE_RATE as i64;
    /// Half of full scale: exactly 0.5 as f32.
    const HALF_SCALE: i32 = 1 << 30;

    fn test_format() -> AudioFormat {
        AudioFormat {
            codec: Codec::Pcm,
            sample_rate: SAMPLE_RATE,
            channels: 2,
            bit_depth: 24,
            codec_header: None,
        }
    }

    /// `ms` milliseconds of a constant half-scale level starting at `timestamp`.
    fn half_scale(timestamp: i64, ms: usize) -> AudioBuffer {
        let samples = vec![HALF_SCALE; ms * SAMPLE_RATE as usize / 1000 * 2];
        AudioBuffer {
            timestamp,
            samples: Arc::from(samples.into_boxed_slice()),
            format: test_format(),
        }
    }

    /// Two exchanges with no offset, so server time is the local clock.
    fn synchronize(clock_sync: &Mutex<ClockSync>, clock: &DefaultClock) {
        let mut sync = clock_sync.lock();
        // The filter only needs increasing receive times; the clock's epoch
        // may be its construction, so stay clear of zero.
        let now = clock.now_micros().max(0) + 1;
        for t1 in [now, now + 1_000] {
            sync.update(t1, t1 + 50, t1 + 50, t1 + 100);
        }
        assert!(sync.is_synchronized());
    }

    struct Harness {
        clock: Arc<DefaultClock>,
        clock_sync: Arc<Mutex<ClockSync>>,
        source: SyncedSource,
        renderer: SyncedRenderer,
        out: Vec<f32>,
    }

    impl Harness {
        fn new(config: SyncedRendererConfig) -> Self {
            Self::build(|format, clock_sync| SyncedRenderer::new(format, clock_sync, config))
        }

        fn build(
            create: impl FnOnce(
                AudioFormat,
                Arc<Mutex<ClockSync>>,
            )
                -> Result<(SyncedSource, SyncedRenderer), crate::error::Error>,
        ) -> Self {
            let clock = Arc::new(DefaultClock::new());
            let clock_sync = Arc::new(Mutex::new(ClockSync::new(clock.clone())));
            let (source, renderer) = create(test_format(), Arc::clone(&clock_sync)).unwrap();
            Self {
                clock,
                clock_sync,
                source,
                renderer,
                out: vec![f32::NAN; FRAMES * 2],
            }
        }

        /// Render one buffer whose first frame is heard at `heard_at`
        /// (server time, which is the local clock once synchronized).
        fn render_heard_at(&mut self, heard_at: i64) -> &[f32] {
            let delay_us = (heard_at - self.clock.now_micros()).max(0) as u64;
            self.renderer
                .render(&mut self.out, Duration::from_micros(delay_us));
            &self.out
        }

        /// Render the next `buffers` buffers of a gapless stream heard from
        /// `*heard_at`, advancing it, and return the last one.
        fn render_stream(&mut self, heard_at: &mut i64, buffers: usize) -> &[f32] {
            for _ in 0..buffers {
                self.render_heard_at(*heard_at);
                *heard_at += FRAMES as i64 * FRAME_US;
            }
            &self.out
        }
    }

    fn is_silent(out: &[f32]) -> bool {
        out.iter().all(|&sample| sample == 0.0)
    }

    #[test]
    fn renders_silence_until_queued_audio_is_due() {
        let mut h = Harness::new(SyncedRendererConfig::new());
        let now = h.clock.now_micros();

        // Nothing queued.
        assert!(is_silent(h.render_heard_at(now)));

        // Audio due in 200ms, but the clock cannot place it yet.
        let start = now + 200_000;
        h.source.enqueue(half_scale(start, 1_000));
        assert!(is_silent(h.render_heard_at(now)));

        // Synchronized: the first render anchors the cursor and stays silent
        // for its own period, then the start gate holds silence while the
        // audio is still ahead of when the output would play it.
        synchronize(&h.clock_sync, &h.clock);
        let now = h.clock.now_micros();
        assert!(is_silent(h.render_heard_at(now)));
        assert!(is_silent(h.render_heard_at(now + 10_000)));

        // A buffer heard when the audio is due plays it from its first frame.
        let mut heard_at = start;
        let out = h.render_stream(&mut heard_at, 1);
        assert!(out.iter().all(|&sample| sample == 0.5), "{out:?}");
        let out = h.render_stream(&mut heard_at, 1);
        assert!(out.iter().all(|&sample| sample == 0.5), "{out:?}");
    }

    #[test]
    fn smooth_correction_plays_in_sync_audio_unchanged() {
        let mut h = Harness::new(SyncedRendererConfig {
            correction: CorrectionMode::Smooth,
            ..SyncedRendererConfig::new()
        });
        synchronize(&h.clock_sync, &h.clock);
        let start = h.clock.now_micros() + 50_000;
        h.source.enqueue(half_scale(start, 1_000));
        h.render_heard_at(h.clock.now_micros());

        let mut heard_at = start;
        for _ in 0..10 {
            let out = h.render_stream(&mut heard_at, 1);
            assert!(out.iter().all(|&sample| sample == 0.5), "{out:?}");
        }
    }

    #[test]
    fn source_volume_and_mute_reach_rendered_output() {
        let mut h = Harness::new(SyncedRendererConfig {
            muted: true,
            ..SyncedRendererConfig::new()
        });
        synchronize(&h.clock_sync, &h.clock);
        let start = h.clock.now_micros() + 50_000;
        h.source.enqueue(half_scale(start, 1_000));
        h.render_heard_at(h.clock.now_micros());

        // Muted from the first buffer, with no ramp from a default gain.
        let mut heard_at = start;
        assert!(is_silent(h.render_stream(&mut heard_at, 1)));

        // Unmuting ramps up over 20ms (two buffers), then plays at unity.
        h.source.set_mute(false);
        let ramp = h.render_stream(&mut heard_at, 1).to_vec();
        assert!(ramp[0] > 0.0 && ramp[0] < ramp[ramp.len() - 1], "{ramp:?}");
        let out = h.render_stream(&mut heard_at, 2);
        assert!(out.iter().all(|&sample| sample == 0.5), "{out:?}");

        // A volume change through the source settles at its gain.
        h.source.set_volume(50);
        let gain = h.source.gain_control().gain();
        assert!(gain > 0.0 && gain < 1.0);
        let out = h.render_stream(&mut heard_at, 3);
        assert!(
            out.iter().all(|&sample| (sample - 0.5 * gain).abs() < 1e-6),
            "{out:?}"
        );

        // A clone controls the same renderer.
        h.source.clone().set_mute(true);
        assert!(h.source.is_muted());
        assert!(is_silent(h.render_stream(&mut heard_at, 3)));
    }

    #[test]
    fn process_callback_output_is_final_including_silence() {
        let calls = Arc::new(AtomicU64::new(0));
        let calls_in_callback = Arc::clone(&calls);
        let mut h = Harness::build(|format, clock_sync| {
            SyncedRenderer::with_process_callback(
                format,
                clock_sync,
                SyncedRendererConfig::new(),
                Box::new(move |data| {
                    calls_in_callback.fetch_add(1, Ordering::Relaxed);
                    for sample in data {
                        *sample = -*sample;
                    }
                }),
            )
        });

        // Called for pre-start silence too.
        assert!(is_silent(h.render_heard_at(h.clock.now_micros())));
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        synchronize(&h.clock_sync, &h.clock);
        let start = h.clock.now_micros() + 50_000;
        h.source.enqueue(half_scale(start, 1_000));
        h.render_heard_at(h.clock.now_micros());
        let mut heard_at = start;
        let out = h.render_stream(&mut heard_at, 1);
        assert!(out.iter().all(|&sample| sample == -0.5), "{out:?}");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn renderer_moves_to_the_output_thread_and_source_is_shared() {
        fn assert_send<T: Send>() {}
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send::<SyncedRenderer>();
        assert_send_sync::<SyncedSource>();
    }

    #[test]
    fn rejects_zero_channels() {
        let format = AudioFormat {
            channels: 0,
            ..test_format()
        };
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));
        assert!(
            SyncedRenderer::new(format, clock_sync, SyncedRendererConfig::new()).is_err(),
            "channels=0 should be rejected"
        );
    }
}
