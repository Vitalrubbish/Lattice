fn main() {
    // WSL2: the real libcuda.so lives in /usr/lib/wsl/lib, not the standard
    // x86_64-linux-gnu path.  Without this rpath the binary loads the wrong
    // stub and fails with CUDA_ERROR_NO_DEVICE.
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
}
