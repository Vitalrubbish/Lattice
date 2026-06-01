fn main() {
    // WSL2: the real libcuda.so lives in /usr/lib/wsl/lib.
    // Only emit the rpath when the WSL2 lib actually exists.
    if std::path::Path::new("/usr/lib/wsl/lib/libcuda.so.1.1").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    }

    // cuFile (GDS) library
    if std::env::var("CARGO_FEATURE_GDS").is_ok() {
        // Locate libcufile.so in CUDA toolkit directories.
        let cufile_paths = &[
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
        ];
        for p in cufile_paths {
            if std::path::Path::new(p).join("libcufile.so").exists() {
                println!("cargo:rustc-link-search=native={p}");
                println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
                break;
            }
        }
    }
}
