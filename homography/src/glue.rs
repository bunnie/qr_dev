use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Pixel, Rgb, Rgba};
use super::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MonoColor(ColorNative);
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mono {
    Black,
    White,
}
impl From<ColorNative> for Mono {
    fn from(value: ColorNative) -> Self {
        match value.0 {
            0 => Mono::Black,
            _ => Mono::White,
        }
    }
}
impl Into<ColorNative> for Mono {
    fn into(self) -> ColorNative {
        match self {
            Mono::Black => ColorNative::from(0),
            Mono::White => ColorNative::from(255),
        }
    }
}
impl From<ColorNative> for Rgb<u8> {
    // alpha | r | g | b
    fn from(value: ColorNative) -> Self {
        Rgb([((value.0 & 0xFF0000) >> 16) as u8, ((value.0 & 0xFF00) >> 8) as u8, (value.0 & 0xFF) as u8])
    }
}
impl Into<ColorNative> for Rgb<u8> {
    fn into(self) -> ColorNative {
        ColorNative((self.0[0] as usize) << 16 | (self.0[1] as usize) << 8 | (self.0[0] as usize))
    }
}

pub fn line(fb: &mut ImageBuffer::<Rgb<u8>, Vec::<u8>>, l: Line, clip: Option<Rectangle>, xor: bool) {
    let color: ColorNative;
    if l.style.stroke_color.is_some() {
        color = l.style.stroke_color.unwrap();
    } else {
        return;
    }
    let mut x0 = l.start.x;
    let mut y0 = l.start.y;
    let x1 = l.end.x;
    let y1 = l.end.y;

    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -((y1 - y0).abs());
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy; // error value e_xy
    loop {
        if x0 >= 0 && y0 >= 0 && x0 < (fb.dimensions().0 as _) && y0 < (fb.dimensions().1 as _) {
            if clip.is_none() || (clip.unwrap().intersects_point(Point::new(x0, y0))) {
                if !xor {
                    fb.put_pixel(x0 as _, y0 as _, color.into());
                } else {
                    if let Some(existing) = fb.get_pixel_checked(x0 as _, y0 as _) {
                        if existing.to_luma().0[0] < 128 {
                            fb.put_pixel(x0 as _, y0 as _, Luma([255]).to_rgb());
                        } else {
                            fb.put_pixel(x0 as _, y0 as _, Luma([0]).to_rgb());
                        }
                    }
                }
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}
