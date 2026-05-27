fn main() {
    // WSL2: the real libcuda.so lives in /usr/lib/wsl/lib.
    // Only emit the rpath when the WSL2 lib actually exists.
    if std::path::Path::new("/usr/lib/wsl/lib/libcuda.so.1.1").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    }
}
