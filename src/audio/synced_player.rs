// ABOUTME: Synced audio player with drift correction
// ABOUTME: Uses DAC callback timestamps to drop/insert frames for alignment

use crate::audio::gain::GainControl;
pub use crate::audio::synced_renderer::{ProcessCallback, MAX_STATIC_DELAY_MS};
use crate::audio::synced_renderer::{SyncedRenderer, SyncedRendererConfig, SyncedSource};
use crate::audio::{AudioBuffer, AudioFormat};
use crate::error::Error;
use crate::sync::ClockSync;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use cpal::{Sample, I24, U24};
use parking_lot::Mutex;
use std::sync::Arc;

/// Configuration for [`SyncedPlayer`] construction.
pub struct SyncedPlayerConfig {
    /// Audio output device. Uses the platform default output device when `None`.
    pub device: Option<Device>,
    /// Initial playback volume, 0-100.
    pub volume: u8,
    /// Initial mute state.
    pub muted: bool,
    /// Optional fixed cpal buffer size in frames. When `None`, Windows
    /// requests a 40ms endpoint buffer (see `WINDOWS_DEFAULT_BUFFER_MS` for
    /// the rationale) and other platforms use the cpal device default.
    pub buffer_size: Option<u32>,
}

impl SyncedPlayerConfig {
    /// Create a config with common playback defaults.
    pub fn new() -> Self {
        Self {
            device: None,
            volume: 100,
            muted: false,
            buffer_size: None,
        }
    }
}

impl Default for SyncedPlayerConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Endpoint buffer requested on Windows when the caller does not override
/// [`SyncedPlayerConfig::buffer_size`].
///
/// WASAPI's shared-mode default allocates two engine periods (20ms at the
/// common 10ms period), so one late wake of the event-loop thread drains the
/// buffer completely and the next slip starves the mixer (silence insertion —
/// a real timeline displacement). Requesting ~40ms deepens the endpoint
/// queue only — the engine still wakes us every period — leaving roughly
/// three periods queued at each wake, so the mixer reliably survives two
/// consecutive late wakes (a third lands exactly at empty). That margin
/// matters even with MMCSS: the thread
/// shares the real-time class with every other pro-audio client and will
/// sometimes lose its slot.
///
/// Depth does not affect inter-device sync: the sync error is measured
/// against padding-compensated presentation timestamps, so queued depth
/// cancels out of the alignment math. The cost is bounded reaction latency —
/// a track-skip flush, volume ramp, or drift correction reaches the speaker
/// only after the already-queued frames drain.
const WINDOWS_DEFAULT_BUFFER_MS: u32 = 40;

/// Frames for [`WINDOWS_DEFAULT_BUFFER_MS`] at `sample_rate`.
const fn windows_default_buffer_frames(sample_rate: u32) -> u32 {
    sample_rate * WINDOWS_DEFAULT_BUFFER_MS / 1000
}

/// Endpoint buffer request when the caller does not specify one.
fn default_buffer_size(sample_rate: u32) -> cpal::BufferSize {
    if cfg!(target_os = "windows") {
        cpal::BufferSize::Fixed(windows_default_buffer_frames(sample_rate))
    } else {
        cpal::BufferSize::Default
    }
}

/// Synced audio output with drift correction.
///
/// Plays through a cpal output stream, running a [`SyncedRenderer`] in its
/// data callback. To play through an output cpal does not drive, use a
/// [`SyncedRenderer`] directly.
pub struct SyncedPlayer {
    source: SyncedSource,
    _stream: Stream,
    /// Last error from the audio stream callback, if any.
    last_error: Arc<Mutex<Option<String>>>,
}

impl SyncedPlayer {
    /// Create a new synced player using the provided clock sync and optional device.
    ///
    /// The player starts at `volume` (0-100) and `muted` state. These are
    /// applied immediately — the first audio callback uses the correct gain
    /// with no ramp from a default value.
    /// The `buffer_size` overrides the endpoint buffer request. If not set,
    /// Windows requests 40ms of margin (see `WINDOWS_DEFAULT_BUFFER_MS` for
    /// the rationale) and other platforms use the cpal device default.
    pub fn new(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, config, None)
    }

    /// Create a player with a process callback for post-gain audio processing.
    ///
    /// The callback receives samples **after** gain/mute processing has been
    /// applied. See [`ProcessCallback`] for thread-safety requirements.
    ///
    /// # Example (requires physical audio hardware to run)
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use parking_lot::Mutex;
    /// # use sendspin::audio::{AudioFormat, Codec, SyncedPlayer, SyncedPlayerConfig};
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
    /// let player = SyncedPlayer::with_process_callback(
    ///     format,
    ///     clock_sync,
    ///     SyncedPlayerConfig::new(),
    ///     Box::new(|data| { /* e.g. feed a VU meter or visualizer */ }),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_process_callback(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        callback: ProcessCallback,
    ) -> Result<Self, Error> {
        Self::build(format, clock_sync, config, Some(callback))
    }

    fn build(
        format: AudioFormat,
        clock_sync: Arc<Mutex<ClockSync>>,
        config: SyncedPlayerConfig,
        process_callback: Option<ProcessCallback>,
    ) -> Result<Self, Error> {
        // Rejects an unusable format before any device access.
        let (source, renderer) = SyncedRenderer::build(
            format.clone(),
            clock_sync,
            SyncedRendererConfig {
                volume: config.volume,
                muted: config.muted,
            },
            process_callback,
        )?;
        let host = cpal::default_host();
        let device = match config.device {
            Some(device) => device,
            None => host
                .default_output_device()
                .ok_or_else(|| Error::Output("No output device available".to_string()))?,
        };

        let stream_config = StreamConfig {
            channels: format.channels as u16,
            sample_rate: cpal::SampleRate::from(format.sample_rate),
            buffer_size: match config.buffer_size {
                Some(frames) => cpal::BufferSize::Fixed(frames),
                None => default_buffer_size(format.sample_rate),
            },
        };

        let last_error = Arc::new(Mutex::new(None));

        let stream = Self::build_stream(
            &device,
            &stream_config,
            &format,
            renderer,
            Arc::clone(&last_error),
        )?;
        stream.play().map_err(|e| Error::Output(e.to_string()))?;
        log::info!(
            "SyncedPlayer started: {} channels, {} Hz, {}-bit",
            format.channels,
            format.sample_rate,
            format.bit_depth,
        );

        Ok(Self {
            source,
            _stream: stream,
            last_error,
        })
    }

    /// Enqueue a decoded buffer for playback.
    ///
    /// Scheduling uses `buffer.timestamp` (server time in microseconds) for
    /// drift-corrected playback.
    pub fn enqueue(&self, buffer: AudioBuffer) {
        self.source.enqueue(buffer);
    }

    /// Clear queued audio and reset playback state.
    pub fn clear(&self) {
        self.source.clear();
    }

    /// Release the underlying audio output device.
    ///
    /// Consuming the player drops its active `cpal::Stream`, allowing another
    /// local source to open the same device. Use this before advertising
    /// `state: "external_source"`.
    ///
    /// The returned [`GainControl`] preserves Sendspin's software volume/mute
    /// state across the handoff. It does not reflect hardware or OS mixer
    /// changes made by the external source.
    pub fn release_audio_device(self) -> GainControl {
        self.source.gain_control().clone()
    }

    /// Return the configured audio format.
    pub fn format(&self) -> &AudioFormat {
        self.source.format()
    }

    /// Check if the audio stream has encountered an error.
    ///
    /// Returns the error message if one occurred, clearing it in the process.
    pub fn take_error(&self) -> Option<String> {
        self.last_error.lock().take()
    }

    /// Check if the audio stream has an error without clearing it.
    pub fn has_error(&self) -> bool {
        self.last_error.lock().is_some()
    }

    /// Get a reference to the volume/mute control.
    ///
    /// Call `.clone()` if you need an owned handle to share across threads
    /// (cloning is cheap — single `Arc` increment, no data copy).
    pub fn gain_control(&self) -> &GainControl {
        self.source.gain_control()
    }

    // -- Volume/mute convenience methods --
    //
    // These promote the most common operations for ergonomics in simple
    // use-cases. For full control, use `gain_control()` directly.

    /// Current volume as 0-100.
    pub fn volume(&self) -> u8 {
        self.gain_control().volume()
    }

    /// Whether playback is currently muted.
    pub fn is_muted(&self) -> bool {
        self.gain_control().is_muted()
    }

    /// Set playback volume (0-100).
    pub fn set_volume(&self, volume: u8) {
        self.gain_control().set_volume(volume);
    }

    /// Set mute state.
    pub fn set_mute(&self, muted: bool) {
        self.gain_control().set_mute(muted);
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
        self.source.set_static_delay(delay_ms);
    }

    /// Current static delay in milliseconds.
    pub fn static_delay_ms(&self) -> u16 {
        self.source.static_delay_ms()
    }

    fn build_stream(
        device: &Device,
        config: &StreamConfig,
        format: &AudioFormat,
        mut renderer: SyncedRenderer,
        error: Arc<Mutex<Option<String>>>,
    ) -> Result<Stream, Error> {
        let mut f32_buffer = Vec::<f32>::new();
        let device_config = device
            .default_output_config()
            .map_err(|e| Error::Output(e.to_string()))?;
        let mut stream_config = device_config.config();
        stream_config.buffer_size = config.buffer_size;
        stream_config.channels = format.channels.into();
        stream_config.sample_rate = format.sample_rate;

        macro_rules! output_stream {
            ($sample:ty) => {
                device
                    .build_output_stream(
                        stream_config,
                        move |data: &mut [$sample], info: &cpal::OutputCallbackInfo| {
                            // How long until the first frame of `data` is heard.
                            let ts = info.timestamp();
                            let playback_delta = ts.playback.duration_since(ts.callback);

                            f32_buffer.resize(data.len(), 0.0);
                            renderer.render(&mut f32_buffer, playback_delta);
                            for (dst, &sample) in data.iter_mut().zip(f32_buffer.iter()) {
                                *dst = <$sample>::from_sample(sample);
                            }
                        },
                        move |err| {
                            // cpal reports a refused real-time promotion as
                            // RealtimeDenied ("Audio will still play"); playback
                            // continues at normal priority. Warn without storing:
                            // take_error()/has_error() signal fatal stream
                            // failures, and a consumer must not tear down a
                            // working stream over a scheduling downgrade.
                            if err.kind() == cpal::ErrorKind::RealtimeDenied {
                                log::warn!(
                                    "Audio thread priority promotion failed (non-fatal): {err}"
                                );
                                return;
                            }
                            log::error!("Audio stream error: {err}");
                            *error.lock() = Some(err.to_string());
                        },
                        None,
                    )
                    .map_err(|e| Error::Output(e.to_string()))
            };
        }

        log::debug!(
            "Using output device: {}, config: {:?}",
            device
                .id()
                .map(|id| format!("{:?}", id))
                .map_err(|e| Error::Output(e.to_string()))?,
            device_config,
        );
        match device_config.sample_format() {
            SampleFormat::F32 => output_stream!(f32),
            SampleFormat::F64 => output_stream!(f64),
            SampleFormat::I8 => output_stream!(i8),
            SampleFormat::I16 => output_stream!(i16),
            SampleFormat::I24 => output_stream!(I24),
            SampleFormat::I32 => output_stream!(i32),
            SampleFormat::I64 => output_stream!(i64),
            SampleFormat::U8 => output_stream!(u8),
            SampleFormat::U16 => output_stream!(u16),
            SampleFormat::U24 => output_stream!(U24),
            SampleFormat::U32 => output_stream!(u32),
            SampleFormat::U64 => output_stream!(u64),
            _ => Err(Error::Output(format!(
                "Unsupported sample format: {:?}",
                device_config.sample_format()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    // Note: SyncedPlayer's convenience methods (volume, is_muted, set_volume,
    // set_mute, gain_control) delegate to GainControl which is thoroughly tested
    // in gain.rs. Wiring tests require a real audio device (cpal Stream) and
    // cannot run in CI.

    use super::{windows_default_buffer_frames, SyncedPlayer, SyncedPlayerConfig};
    use crate::audio::{AudioFormat, Codec};
    use crate::error::Error;
    use crate::sync::{ClockSync, DefaultClock};
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[test]
    fn test_build_rejects_zero_channels() {
        // `SyncedPlayer::build` short-circuits on channels==0 *before* any
        // cpal device access, so this test works regardless of whether the
        // runner has audio hardware. Asserting on the specific error message
        // (not just `is_err()`) pins the check to the explicit guard — any
        // later failure (missing device, cpal rejecting the config) would
        // surface a different message.
        let format = AudioFormat {
            codec: Codec::Pcm,
            sample_rate: 48_000,
            channels: 0,
            bit_depth: 24,
            codec_header: None,
        };
        let clock_sync = Arc::new(Mutex::new(ClockSync::new(Arc::new(DefaultClock::new()))));
        let result = SyncedPlayer::new(format, clock_sync, SyncedPlayerConfig::new());
        let err = match result {
            Ok(_) => panic!("channels=0 should be rejected"),
            Err(e) => e,
        };
        match err {
            Error::Output(msg) => assert!(
                msg.contains("channels must be > 0"),
                "expected 'channels must be > 0' error, got: {msg}"
            ),
            other => panic!("expected Error::Output, got {other:?}"),
        }
    }

    #[test]
    fn test_windows_default_buffer_frames_is_40ms() {
        assert_eq!(windows_default_buffer_frames(44_100), 1_764);
        assert_eq!(windows_default_buffer_frames(48_000), 1_920);
        assert_eq!(windows_default_buffer_frames(192_000), 7_680);
    }
}
