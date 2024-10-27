mod homography;
use homography::*;
mod qr;
use qr::*;
mod minigfx;
use minigfx::*;
mod glue;
use image::Pixel;

use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use minifb::{Window, WindowOptions, Key};

fn show_image(img: &DynamicImage) {
    // Get image dimensions and raw pixel data
    let (width, height) = img.dimensions();

    let img_data = img.to_rgba8();
    let buffer: Vec<u32> = img_data
        .chunks(4)
        .map(|pixel| {
            let [r, g, b, _a] = [pixel[0], pixel[1], pixel[2], pixel[3]];
            (r as u32) << 16 | (g as u32) << 8 | (b as u32)
        })
        .collect();

    // Create a window
    let mut window = Window::new(
        "PNG Viewer",
        width as usize,
        height as usize,
        WindowOptions::default(),
    )
    .expect("Failed to create window");

    // Display the image
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window
            .update_with_buffer(&buffer, width as usize, height as usize)
            .expect("Failed to update buffer");
    }
}

fn main() {
    // Load the PNG file
    let img = image::open("../images/test256b.png").expect("Failed to load image");

    // Get image dimensions and raw pixel data
    let (width, _height) = img.dimensions();
    let mut candidates: [Option::<Point>; 8] = [None; 8];
    let image = img.clone().into_luma8();
    let hom_src = img.clone().into_luma8();
    find_finders(&mut candidates, &image, 128, width as _);

    let mut drawable_image = img.to_rgb8();

    const CROSSHAIR_LEN: isize = 3;
    let mut candidates_found = 0;
    let mut candidate3 = [Point::new(0, 0); 3];
    for candidate in candidates.iter() {
        if let Some(c) = candidate {
            if candidates_found < candidate3.len() {
                candidate3[candidates_found] = *c;
            }
            candidates_found += 1;
            println!("******    candidate: {}, {}    ******", c.x, c.y);
        }
    }

    if candidates_found == 3 {
        // Example usage with known three points
        let p1 = candidate3[0].to_f32();
        let p2 = candidate3[1].to_f32();
        let p3 = candidate3[2].to_f32();

        // Determine the fourth point
        let p4 = find_fourth_point(p1, p2, p3);
        println!("The fourth point is: {:?}", p4);
        let dst = [(24.0f32, 24.0f32), (216.0f32, 24.0f32), (216.0f32, 216.0f32), (24.0f32, 216.0f32)];
        let src = [p1, p2, p3, (41.0 + 9.0, 192.0 - 2.0)];
        for p in src {
            let c_screen = Point::new(p.0 as isize, p.1 as isize);
            // flip coordinates to match the camera data
            // c_screen = Point::new(c_screen.x, drawable_image.dimensions().1 as isize - 1 - c_screen.y);
            // vertical cross hair
            glue::line(
                &mut drawable_image,
                Line::new_with_style(
                    c_screen + Point::new(0, CROSSHAIR_LEN),
                    c_screen - Point::new(0, CROSSHAIR_LEN),
                    DrawStyle::stroke_color(Rgb([0,255,0]).into())
                ),
                None,
                false,
            );
            // horizontal cross hair
            glue::line(
                &mut drawable_image,
                Line::new_with_style(
                    c_screen + Point::new(CROSSHAIR_LEN, 0),
                    c_screen - Point::new(CROSSHAIR_LEN, 0),
                    DrawStyle::stroke_color(Rgb([0,255,0]).into())
                ),
                None,
                false,
            );
        }
        show_image(&DynamicImage::ImageRgb8(drawable_image));

        let mut dest_img = RgbImage::new(hom_src.dimensions().0, hom_src.dimensions().1);
        match find_homography(src, dst) {
            Some(h) => {
                println!("{:}", h);
                // iterate through pixels and apply homography
                for y in 0..hom_src.dimensions().1 {
                    for x in 0..hom_src.dimensions().0 {
                        let (x_dst, y_dst) = apply_homography(&h, (x as f32, y as f32));
                        if (x_dst as i32 >= 0) && ((x_dst as i32) < hom_src.dimensions().0 as i32)
                        && (y_dst as i32 >= 0) && ((y_dst as i32) < hom_src.dimensions().1 as i32) {
                            // println!("{},{} -> {},{}", x_src as i32, y_src as i32, x, y);
                            dest_img.put_pixel(x_dst as u32, y_dst as u32, hom_src.get_pixel(x as u32, y as u32).to_rgb());
                        }

                        let (x_dst, y_dst) = apply_homography(&h, (x as f32 + 0.5, y as f32));
                        if (x_dst as i32 >= 0) && ((x_dst as i32) < hom_src.dimensions().0 as i32)
                        && (y_dst as i32 >= 0) && ((y_dst as i32) < hom_src.dimensions().1 as i32) {
                            // println!("{},{} -> {},{}", x_src as i32, y_src as i32, x, y);
                            dest_img.put_pixel(x_dst as u32, y_dst as u32, hom_src.get_pixel(x as u32, y as u32).to_rgb());
                        }

                        let (x_dst, y_dst) = apply_homography(&h, (x as f32, y as f32 + 0.5));
                        if (x_dst as i32 >= 0) && ((x_dst as i32) < hom_src.dimensions().0 as i32)
                        && (y_dst as i32 >= 0) && ((y_dst as i32) < hom_src.dimensions().1 as i32) {
                            // println!("{},{} -> {},{}", x_src as i32, y_src as i32, x, y);
                            dest_img.put_pixel(x_dst as u32, y_dst as u32, hom_src.get_pixel(x as u32, y as u32).to_rgb());
                        }

                        let (x_dst, y_dst) = apply_homography(&h, (x as f32 + 0.5, y as f32 + 0.5));
                        if (x_dst as i32 >= 0) && ((x_dst as i32) < hom_src.dimensions().0 as i32)
                        && (y_dst as i32 >= 0) && ((y_dst as i32) < hom_src.dimensions().1 as i32) {
                            // println!("{},{} -> {},{}", x_src as i32, y_src as i32, x, y);
                            dest_img.put_pixel(x_dst as u32, y_dst as u32, hom_src.get_pixel(x as u32, y as u32).to_rgb());
                        }
                    }
                }
                show_image(&DynamicImage::ImageRgb8(dest_img));
            }
            _ => println!("err")
        }
    }
}
