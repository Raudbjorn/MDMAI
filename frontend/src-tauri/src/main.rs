//! Main entry point for the MDMAI Desktop Application
//! 
//! This file serves as the main entry point for the Tauri desktop application.
//! It sets up the application environment and starts the main event loop.

// Prevents additional console window on Windows in release builds
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    mdmai_desktop::run_app();
}