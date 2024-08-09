// This script was copy/pasted from https://github.com/tracel-ai/burn/blob/main/xtask/src/main.rs
use clap::{Parser, Subcommand};

mod publish;
mod utils;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Publish a crate to crates.io
    Publish {
        /// The name of the crate to publish on crates.io
        name: String,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Publish { name } => publish::run(name),
    }
}
