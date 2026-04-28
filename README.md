# jxl-rs

An idiomatic Rust JPEG XL implementation in progress.

The current codebase is deliberately focused on the foundation every conforming
decoder needs:

- JPEG XL naked codestream and container signature detection.
- JPEG XL container box parsing, including extended-size boxes.
- Codestream extraction from `jxlc` and ordered `jxlp` boxes.
- JPEG XL size-header parsing with the same field encoding as libjxl.
- Top-level image metadata parsing: orientation, intrinsic/preview/animation
  flags, bit depth, extra channels, XYB/original-profile flag, color encoding,
  and tone mapping.
- Custom transform-data parsing after basic info, including optional custom
  opsin inverse matrix and upsampling kernel weights.
- ICC profile parsing for codestreams that signal `want_icc`, including the
  JPEG XL ICC entropy stream and predictor reversal.
- First-frame header parsing: frame type, modular/VarDCT encoding selection,
  color transform, frame crop/origin, passes, blending, animation timing,
  loop-filter parameters, and group layout.
- First-frame data table-of-contents parsing and section payload traversal,
  including DC/global and AC group section classification.
- First-frame modular global metadata parsing for frames without optional
  patch/spline/noise preambles, including the DC dequant preamble, global MA
  tree, weighted-predictor header, and modular transforms.
- First-frame modular DC/AC group stream header parsing, including stream ID
  derivation and local MA tree metadata when a group does not use the global
  tree.
- First-frame modular channel planning after global metadata transforms, with
  per-group channel rectangles for future residual token decoding.
- A small public inspection API through the `jxl` crate.
- A `jxlinfo-rs` CLI for metadata inspection.
- Fixture tests against `reference/libjxl/testdata`, with optional comparison
  to the built libjxl `jxlinfo` reference tool.

Pixel reconstruction is not implemented yet. The next decoder slices should add
modular residual token decoding, modular inverse transforms, and then VarDCT.

## Workspace

- `crates/jxl-codec`: core codestream and container primitives.
- `crates/jxl`: public Rust-native API.
- `crates/jxl-cli`: command-line tools.
- `reference/libjxl`: C++ reference implementation used only as an oracle in
  tests and development, never through FFI.

## Usage

```sh
cargo run -p jxl-cli --bin jxlinfo-rs -- reference/libjxl/testdata/jxl/splines.jxl
```

## Reference Tools

The tests can compare parsed dimensions against the reference `jxlinfo` tool.
The default path is:

```text
reference/libjxl/build-rs-oracle/tools/jxlinfo
```

To rebuild the reference tools:

```sh
cmake -S reference/libjxl -B reference/libjxl/build-rs-oracle -G Ninja \
  -DJPEGXL_ENABLE_PLUGINS=OFF \
  -DJPEGXL_ENABLE_MANPAGES=OFF \
  -DJPEGXL_ENABLE_BENCHMARK=OFF \
  -DJPEGXL_ENABLE_EXAMPLES=OFF \
  -DJPEGXL_ENABLE_JNI=OFF \
  -DJPEGXL_ENABLE_DEVTOOLS=OFF \
  -DJPEGXL_ENABLE_VIEWERS=OFF
cmake --build reference/libjxl/build-rs-oracle --target jxlinfo djxl cjxl -j 8
```

Set `JXL_RS_REFERENCE_JXLINFO=/path/to/jxlinfo` to use a different oracle
binary.
