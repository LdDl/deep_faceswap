use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use deep_faceswap_core::utils::blur::gaussian_blur_2d;
use deep_faceswap_core::utils::cv::{erode_mask, erode_mask_optimized};
use ndarray::Array2;

fn bench_gaussian_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur");

    let sizes = vec![
        // Small image, medium kernel
        (100, 100, 15),
        // Large image, medium kernel
        (640, 640, 15),
        // Large image, large kernel (paste_back case)
        (640, 640, 65),
    ];

    for (h, w, kernel) in sizes {
        let input = Array2::<f32>::from_elem((h, w), 128.0);

        group.bench_with_input(
            BenchmarkId::new("gaussian_blur", format!("{}x{}_k{}", h, w, kernel)),
            &(&input, kernel),
            |b, (input, kernel)| b.iter(|| gaussian_blur_2d(black_box(input), black_box(*kernel))),
        );
    }

    group.finish();
}

fn bench_erosion(c: &mut Criterion) {
    let mut group = c.benchmark_group("erosion");

    let sizes = vec![
        // Small image, small kernel
        (100, 100, 3),
        // Large image, paste_back typical kernel
        (640, 640, 11),
        // Large image, large kernel
        (640, 640, 21),
    ];

    for (h, w, kernel) in sizes {
        let input = Array2::<u8>::from_elem((h, w), 255);

        group.bench_with_input(
            BenchmarkId::new("original", format!("{}x{}_k{}", h, w, kernel)),
            &(&input, kernel),
            |b, (input, kernel)| b.iter(|| erode_mask(black_box(input), black_box(*kernel))),
        );

        group.bench_with_input(
            BenchmarkId::new("optimized", format!("{}x{}_k{}", h, w, kernel)),
            &(&input, kernel),
            |b, (input, kernel)| {
                b.iter(|| erode_mask_optimized(black_box(input), black_box(*kernel)))
            },
        );
    }

    group.finish();
}

fn bench_paste_back_combined(c: &mut Criterion) {
    let mut group = c.benchmark_group("paste_back_combined");
    group.sample_size(20);

    let mask_f32 = Array2::<f32>::from_elem((640, 640), 128.0);
    let mask_u8 = Array2::<u8>::from_elem((640, 640), 255);

    group.bench_function("original_erosion + original_blur", |b| {
        b.iter(|| {
            let eroded = erode_mask(black_box(&mask_u8), black_box(11));
            let _blurred = gaussian_blur_2d(black_box(&mask_f32), black_box(65));
            eroded
        })
    });

    group.bench_function("optimized_erosion + original_blur", |b| {
        b.iter(|| {
            let eroded = erode_mask_optimized(black_box(&mask_u8), black_box(11));
            let _blurred = gaussian_blur_2d(black_box(&mask_f32), black_box(65));
            eroded
        })
    });

    group.finish();
}

fn bench_roi_vs_fullimage(c: &mut Criterion) {
    let mut group = c.benchmark_group("roi_vs_fullimage");
    group.sample_size(10);

    // Simulate 4K image (3840x2160) with a 350x350 face region
    let full_h = 2160;
    let full_w = 3840;
    let roi_h = 350;
    let roi_w = 350;
    let erosion_k = 11;
    let blur_k = 65;

    // Full-image approach (old paste_back)
    let full_mask_u8 = Array2::<u8>::from_elem((full_h, full_w), 0);
    let full_mask_f32 = Array2::<f32>::from_elem((full_h, full_w), 0.0);

    group.bench_function("full_4K_erosion+blur", |b| {
        b.iter(|| {
            let eroded = erode_mask_optimized(black_box(&full_mask_u8), black_box(erosion_k));
            let eroded_f32 = eroded.mapv(|v| v as f32);
            let _blurred = gaussian_blur_2d(black_box(&eroded_f32), black_box(blur_k));
        })
    });

    // ROI approach (new paste_back_inplace)
    let roi_mask_u8 = Array2::<u8>::from_elem((roi_h, roi_w), 0);
    let roi_mask_f32 = Array2::<f32>::from_elem((roi_h, roi_w), 0.0);

    group.bench_function("roi_350x350_erosion+blur", |b| {
        b.iter(|| {
            let eroded = erode_mask_optimized(black_box(&roi_mask_u8), black_box(erosion_k));
            let eroded_f32 = eroded.mapv(|v| v as f32);
            let _blurred = gaussian_blur_2d(black_box(&eroded_f32), black_box(blur_k));
        })
    });

    // Also measure array allocation difference (24MB vs 384KB)
    group.bench_function("alloc_full_4K_mask", |b| {
        b.iter(|| {
            let _mask = black_box(Array2::<u8>::zeros((full_h, full_w)));
            let _face = black_box(ndarray::Array3::<f32>::zeros((full_h, full_w, 3)));
        })
    });

    group.bench_function("alloc_roi_350x350", |b| {
        b.iter(|| {
            let _mask = black_box(Array2::<u8>::zeros((roi_h, roi_w)));
            let _face = black_box(ndarray::Array3::<f32>::zeros((roi_h, roi_w, 3)));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_gaussian_blur,
    bench_erosion,
    bench_paste_back_combined,
    bench_roi_vs_fullimage
);
criterion_main!(benches);
