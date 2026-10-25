use std::process::Command;
fn main() {
    println!("cargo:rerun-if-env-changed=RPIC_BUILD_REV");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/refs/heads/main");
    let revision = std::env::var("RPIC_BUILD_REV")
        .ok()
        .or_else(|| {
            let result = Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()?;
            if !result.status.success() {
                return None;
            }
            Some(String::from_utf8_lossy(&result.stdout).trim().to_owned())
        })
        .unwrap_or_else(|| "unrecorded".into());
    println!("cargo:rustc-env=RPIC_BUILD_REV={revision}");
}
