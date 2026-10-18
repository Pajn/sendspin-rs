// ABOUTME: Protocol implementation for Sendspin WebSocket protocol
// ABOUTME: Message types, serialization, and WebSocket client

/// WebSocket client implementation
#[cfg(feature = "client")]
pub mod client;
/// Builder for easy construction of the client
#[cfg(feature = "client")]
pub mod client_builder;
/// Inbound WebSocket acceptor for server-initiated connections
#[cfg(feature = "client")]
pub mod listener;
/// Managed connection lifecycle: multi-server arbitration and auto-goodbye
#[cfg(feature = "client")]
pub mod manager;
/// Protocol message type definitions and serialization
pub mod messages;

#[cfg(feature = "client")]
pub use client::{Connection, ConnectionGuard, Controller, WsSender};
#[cfg(feature = "client")]
pub use listener::ProtocolListener;
#[cfg(feature = "client")]
pub use manager::{should_switch, ConnectionManager, ManagedConnection, ManagerConfig};
pub use messages::Message;
