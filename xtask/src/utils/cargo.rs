// This script was copy/pasted from https://github.com/tracel-ai/burn/blob/main/xtask/src/utils/cargo.rs
use std::{
    collections::HashMap,
    path::Path,
    process::{Command, Stdio},
};

use crate::utils::process::handle_child_process;

use super::Params;

/// Run a cargo command
pub(crate) fn run_cargo(command: &str, params: Params, envs: HashMap<&str, String>, error: &str) {
    run_cargo_with_path::<String>(command, params, envs, None, error)
}

/// Run a cargo command with the passed directory as the current directory
pub(crate) fn run_cargo_with_path<P: AsRef<Path>>(
    command: &str,
    params: Params,
    envs: HashMap<&str, String>,
    path: Option<P>,
    error: &str,
) {
    let mut cargo = Command::new("cargo");
    cargo
        .env("CARGO_INCREMENTAL", "0")
        .envs(&envs)
        .arg(command)
        .args(&params.params)
        .stdout(Stdio::inherit()) // Send stdout directly to terminal
        .stderr(Stdio::inherit()); // Send stderr directly to terminal

    if let Some(path) = path {
        cargo.current_dir(path);
    }

    // Handle cargo child process
    let cargo_process = cargo.spawn().expect(error);
    handle_child_process(cargo_process, "Cargo process should run flawlessly");
}
