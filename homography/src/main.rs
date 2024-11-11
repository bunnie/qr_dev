mod homography;

use homography::*;
mod qr;
use qr::*;
mod minigfx;
use minigfx::*;
mod glue;
mod modules;
mod version_db;
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Pixel, Rgb, RgbImage, RgbaImage};
use minifb::{Key, Window, WindowOptions};
use modules::*;

fn show_image(flipped_img: &DynamicImage) {
    // Get image dimensions and raw pixel data
    let (width, height) = flipped_img.dimensions();

    // Flip the coordinate system
    let mut img = RgbaImage::new(flipped_img.dimensions().0, flipped_img.dimensions().1);
    for (x, y, pixel) in flipped_img.clone().into_rgba8().enumerate_pixels() {
        img.put_pixel(x, flipped_img.dimensions().1 - y - 1, *pixel);
    }
    let buffer: Vec<u32> = img
        .chunks(4)
        .map(|pixel| {
            let [r, g, b, _a] = [pixel[0], pixel[1], pixel[2], pixel[3]];
            (r as u32) << 16 | (g as u32) << 8 | (b as u32)
        })
        .collect();

    // Create a window
    let mut window = Window::new("PNG Viewer", width as usize, height as usize, WindowOptions::default())
        .expect("Failed to create window");

    // Display the image
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&buffer, width as usize, height as usize).expect("Failed to update buffer");
    }
}

const BW_THRESH: u8 = 128;
/// Finder search margin, as defined by expected QR code code widths (so this scales with the effective
/// resolution of the code)
const FINDER_SEARCH_MARGIN: isize = 2;
/// How much we want the final QR image to be "pulled in" from the outer edge of the image buffer
const HOMOGRAPHY_MARGIN: isize = -8;
const CROSSHAIR_LEN: isize = 3;

fn draw_crosshair(image: &mut RgbImage, p: Point, color: [u8; 3]) {
    glue::line(
        image,
        Line::new_with_style(
            p + Point::new(0, CROSSHAIR_LEN),
            p - Point::new(0, CROSSHAIR_LEN),
            DrawStyle::stroke_color(Rgb(color).into()),
        ),
        None,
        false,
    );
    glue::line(
        image,
        Line::new_with_style(
            p + Point::new(CROSSHAIR_LEN, 0),
            p - Point::new(CROSSHAIR_LEN, 0),
            DrawStyle::stroke_color(Rgb(color).into()),
        ),
        None,
        false,
    );
}

fn draw_line(image: &mut RgbImage, l: &LineDerivation, color: [u8; 3]) {
    let axis = l.independent_axis;
    let (m, b) = l.equation.unwrap();
    match axis {
        Axis::X => {
            for x in 0..image.width() {
                let y = (x as f32 * m + b) as isize;
                if y >= 0 && y < image.height() as isize {
                    image.put_pixel(x as u32, y as u32, Rgb(color));
                }
            }
        }
        Axis::Y => {
            for y in 0..image.height() {
                let x = (y as f32 * m + b) as isize;
                if x >= 0 && x < image.width() as isize {
                    image.put_pixel(x as u32, y as u32, Rgb(color));
                }
            }
        }
    }
}

fn main() {
    // Load the PNG file
    let mut img = image::open("../images/test256b.png").expect("Failed to load image").into_luma8();
    let dims = img.dimensions();
    println!("dimensions: {:?}", dims);
    // Handle to the luma version for image processing
    let mut image = &mut img;
    // Create an RGB version for UI debugging
    let mut rgb_image = RgbImage::new(dims.0, dims.1);
    for (x, y, pixel) in image.enumerate_pixels() {
        let luma = pixel[0];
        rgb_image.put_pixel(x, y, Rgb([luma, luma, luma]));
    }
    let drawable_image = &mut rgb_image;

    // Get image dimensions and raw pixel data
    let (width, _height) = dims;
    let mut candidates = Vec::<Point>::new();
    let finder_width = qr::find_finders(&mut candidates, &image, BW_THRESH, width as _) as isize;

    for &candidate in candidates.iter() {
        println!("******    candidate: {}, {}    ******", candidate.x, candidate.y);
    }

    if candidates.len() == 3 {
        let candidates_orig = candidates.clone();
        let mut qr_corners = QrCorners::from_finders(
            &candidates.try_into().unwrap(),
            image.dimensions(),
            // add a search margin on the finder width
            (finder_width + (FINDER_SEARCH_MARGIN * finder_width) / (1 + 1 + 3 + 1 + 1)) as usize,
        )
        .expect("Bad arguments to QR code finder");
        let mut il = ImageRoi::new(&mut image, dims, BW_THRESH);
        let (src, dst) = qr_corners.mapping(&mut il, HOMOGRAPHY_MARGIN, drawable_image);
        for s in src.iter() {
            if let Some(p) = s {
                println!("src {:?}", p);
                draw_crosshair(drawable_image, *p, [0, 255, 0]);
            }
        }
        for d in dst.iter() {
            if let Some(p) = d {
                println!("dst {:?}", p);
                draw_crosshair(drawable_image, *p, [255, 0, 0]);
            }
        }

        show_image(&DynamicImage::ImageRgb8(drawable_image.clone()));

        let mut dest_img = GrayImage::new(qr_corners.qr_pixels() as u32, qr_corners.qr_pixels() as u32);
        let mut src_f: [(f32, f32); 4] = [(0.0, 0.0); 4];
        let mut dst_f: [(f32, f32); 4] = [(0.0, 0.0); 4];
        let mut all_found = true;
        for (s, s_f32) in src.iter().zip(src_f.iter_mut()) {
            if let Some(p) = s {
                *s_f32 = p.to_f32();
            } else {
                all_found = false;
            }
        }
        for (d, d_f32) in dst.iter().zip(dst_f.iter_mut()) {
            if let Some(p) = d {
                *d_f32 = p.to_f32();
            } else {
                all_found = false;
            }
        }
        if all_found {
            match find_homography(src_f, dst_f) {
                Some(h) => {
                    if let Some(h_inv) = h.try_inverse() {
                        println!("{:?}", h_inv);
                        let h_inv_fp = matrix3_to_fixp(h_inv);
                        println!("{:?}", h_inv_fp);
                        // iterate through pixels and apply homography
                        for y in 0..qr_corners.qr_pixels() {
                            for x in 0..qr_corners.qr_pixels() {
                                let (x_src, y_src) = apply_fixp_homography(&h_inv_fp, (x as i32, y as i32));
                                if (x_src as i32 >= 0)
                                    && ((x_src as i32) < image.dimensions().0 as i32)
                                    && (y_src as i32 >= 0)
                                    && ((y_src as i32) < image.dimensions().1 as i32)
                                {
                                    // println!("{},{} -> {},{}", x_src as i32, y_src as i32, x, y);
                                    dest_img.put_pixel(
                                        x as u32,
                                        y as u32,
                                        image.get_pixel(x_src as u32, y_src as u32).to_luma(),
                                    );
                                } else {
                                    dest_img.put_pixel(x as u32, y as u32, Luma([255]));
                                }
                            }
                        }

                        // we now have a QR code in "canonical" orientation, with a
                        // known width in pixels
                        let qr_width = qr_corners.qr_pixels();

                        // we can also know the location of the finders by transforming them
                        let mut x_candidates: [Point; 3] = [Point::new(0, 0); 3];
                        let h_fp = matrix3_to_fixp(h);
                        for (i, &c) in candidates_orig.iter().enumerate() {
                            let (x, y) = apply_fixp_homography(&h_fp, (c.x as i32, c.y as i32));
                            x_candidates[i] = Point::new(x as isize, y as isize);
                        }

                        let mut debug_qr_image = RgbImage::new(qr_width as _, qr_width as _);
                        for (x, y, pixel) in dest_img.enumerate_pixels() {
                            let luma = pixel[0];
                            debug_qr_image.put_pixel(x, y, Rgb([luma, luma, luma]));
                        }

                        for &x in x_candidates.iter() {
                            println!("transformed finder location {:?}", x);
                            draw_crosshair(&mut debug_qr_image, x, [0, 255, 0]);
                        }
                        show_image(&DynamicImage::ImageRgb8(debug_qr_image.clone()));

                        // Confirm that the finders coordinates are valid
                        let mut checked_candidates = Vec::<Point>::new();
                        let x_finder_width =
                            qr::find_finders(&mut checked_candidates, &dest_img, BW_THRESH, qr_width as _)
                                as isize;

                        println!("x_finder width: {}", x_finder_width);
                        // check that the new coordinates are within 1 pixel of the original
                        const XFORM_DELTA: isize = 1;
                        let mut deltas = Vec::<Point>::new();
                        for c in checked_candidates {
                            println!("x_point: {:?}", c);
                            for &xformed in x_candidates.iter() {
                                let delta = xformed - c;
                                if delta.x.abs() <= XFORM_DELTA && delta.y.abs() <= XFORM_DELTA {
                                    deltas.push(delta);
                                    println!("delta: {:?}", delta);
                                }
                            }
                        }
                        if deltas.len() != 3 {
                            println!("Transformation did not survive sanity check");
                            return;
                        }
                        let (version, modules) = qr::guess_code_version(x_finder_width as usize, qr_width);

                        println!("width: {}, image dims: {:?}", qr_width, dest_img.dimensions());
                        println!("guessed version: {}, modules: {}", version, modules);

                        #[cfg(feature = "rqrr")]
                        {
                            let decode_img_rgb = DynamicImage::ImageRgb8(dest_img.clone());
                            let decode_img = decode_img_rgb.into_luma8();
                            show_image(&DynamicImage::ImageRgb8(dest_img));

                            let mut search_img = rqrr::PreparedImage::prepare(decode_img);
                            let grids = search_img.detect_grids();
                            println!("grids len {}", grids.len());
                            let rawdata = grids[0].get_raw_data();
                            match rawdata {
                                Ok((md, rd)) => {
                                    println!("{:?}, {}:{:x?}", md, rd.len, &rd.data[..(rd.len / 8) + 1]);
                                }
                                Err(e) => {
                                    println!("Error: {:?}", e);
                                }
                            }
                            println!("{:?}", grids[0].decode());
                        }
                    } else {
                        println!("Matrix is not invertable");
                    }
                }
                _ => println!("err"),
            }
        } else {
            println!("Not all points exist, can't do homography transformation");
        }
    }
}
