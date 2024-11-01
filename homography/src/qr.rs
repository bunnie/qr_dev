const SEQ_LEN: usize = 5;
use core::cell::RefCell;
use std::ops::{BitXor, Not};

use nalgebra::{QR, base, indexing};

use super::*;

// use fixed point for the maths. This defines where we fix the point at.
const SEQ_FP_SHIFT: usize = 4;
// searching for a 1:1:3:1:1 black-white-black-white-black pattern
// upper/lower thresholds for recognizing a "1" in the ratio
const LOWER_1: usize = (1 << SEQ_FP_SHIFT) / 2; // "0.5"
const UPPER_1: usize = 2 << SEQ_FP_SHIFT;
// upper/lower thresholds for recognizing a "3" in the ratio
const LOWER_3: usize = 2 << SEQ_FP_SHIFT;
const UPPER_3: usize = 4 << SEQ_FP_SHIFT;

#[derive(Copy, Clone, Default, Debug)]
pub struct QrCorners {
    north_west: Option<Point>,
    north_east: Option<Point>,
    south_west: Option<Point>,
    south_east: Option<Point>,
}
impl QrCorners {
    pub fn from_finders(points: &[Point; 3], dimensions: (u32, u32)) -> Option<Self> {
        let (widthu32, heightu32) = dimensions;
        let x_half = widthu32 as isize / 2;
        let y_half = heightu32 as isize / 2;

        let mut qrc = QrCorners::default();

        for &p in points {
            if p.x < x_half && p.y < y_half {
                qrc.south_west = Some(p);
            } else if p.x < x_half && p.y >= y_half {
                qrc.north_west = Some(p);
            } else if p.x >= x_half && p.y < y_half {
                qrc.south_east = Some(p);
            } else if p.x >= x_half && p.y >= y_half {
                qrc.north_east = Some(p);
            }
        }

        // check that at least three corners are filled
        if (if qrc.north_west.is_some() { 1 } else { 0 }
            + if qrc.north_east.is_some() { 1 } else { 0 }
            + if qrc.south_west.is_some() { 1 } else { 0 }
            + if qrc.south_east.is_some() { 1 } else { 0 })
            == 3
        {
            Some(qrc)
        } else {
            None
        }
    }

    pub fn missing_corner_direction(&self) -> Option<SearchDirection> {
        if self.north_east.is_none() {
            Some(SearchDirection::NorthEast)
        } else if self.north_west.is_none() {
            Some(SearchDirection::NorthWest)
        } else if self.south_east.is_none() {
            Some(SearchDirection::SouthEast)
        } else if self.south_west.is_none() {
            Some(SearchDirection::SouthWest)
        } else {
            None
        }
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct FinderSeq {
    /// run length of the pixels leading up to the current position
    pub run: usize,
    /// the position
    pub pos: usize,
    /// the luminance of the pixels in the run leading up to the current position
    pub color: Color,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Color {
    Black,
    White,
}
impl Color {
    pub fn from(luminance: u8, thresh: u8) -> Self {
        if luminance > thresh { Color::White } else { Color::Black }
    }
}
/// Used for counting pixels
impl Into<usize> for Color {
    fn into(self) -> usize {
        match self {
            Color::Black => 0,
            Color::White => 1,
        }
    }
}
/// Used for translating pixels into RGB or Luma color spaces
impl Into<u8> for Color {
    fn into(self) -> u8 {
        match self {
            Color::Black => 0,
            Color::White => 255,
        }
    }
}
impl Default for Color {
    fn default() -> Self { Color::Black }
}

impl Not for Color {
    type Output = Color;

    fn not(self) -> Self::Output {
        match self {
            Color::Black => Color::White,
            Color::White => Color::Black,
        }
    }
}

impl BitXor for Color {
    type Output = Color;

    fn bitxor(self, rhs: Self) -> Self::Output { if self == rhs { Color::Black } else { Color::White } }
}

#[derive(Copy, Clone, Debug)]
pub enum SearchDirection {
    North,
    NorthWest,
    NorthEast,
    West,
    East,
    South,
    SouthWest,
    SouthEast,
}
impl Into<Point> for SearchDirection {
    fn into(self) -> Point {
        use SearchDirection::*;
        match self {
            North => Point::new(0, 1),
            West => Point::new(-1, 0),
            East => Point::new(1, 0),
            South => Point::new(0, -1),
            NorthWest => Point::new(-1, 1),
            NorthEast => Point::new(1, 1),
            SouthWest => Point::new(-1, -1),
            SouthEast => Point::new(1, -1),
        }
    }
}

pub struct SearchDirectionIter {
    index: usize,
}

impl SearchDirection {
    pub fn iter() -> SearchDirectionIter { SearchDirectionIter { index: 0 } }
}

impl Iterator for SearchDirectionIter {
    type Item = SearchDirection;

    fn next(&mut self) -> Option<Self::Item> {
        let direction = match self.index {
            0 => Some(SearchDirection::North),
            1 => Some(SearchDirection::NorthWest),
            2 => Some(SearchDirection::NorthEast),
            3 => Some(SearchDirection::West),
            4 => Some(SearchDirection::East),
            5 => Some(SearchDirection::South),
            6 => Some(SearchDirection::SouthWest),
            7 => Some(SearchDirection::SouthEast),
            _ => None,
        };
        self.index += 1;
        direction
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TrendDirection {
    Positive,
    None,
    Negative,
}
impl From<isize> for TrendDirection {
    fn from(value: isize) -> Self {
        if value > 0 {
            TrendDirection::Positive
        } else if value < 0 {
            TrendDirection::Negative
        } else {
            TrendDirection::None
        }
    }
}
impl Into<isize> for TrendDirection {
    fn into(self) -> isize {
        match self {
            TrendDirection::Positive => 1,
            TrendDirection::None => 0,
            TrendDirection::Negative => -1,
        }
    }
}

/// Doesn't need to be too big, because the overall area is at most 256, and
/// we pay a performance penalty to zeroize unused space.
const TREND_LEN: usize = 64;
#[derive(Debug)]
pub struct Trend {
    data: [usize; TREND_LEN],
    index: usize,
    greater: isize,
    lesser: isize,
    last: usize,
}
impl Trend {
    pub fn new() -> Self { Self { data: [0usize; TREND_LEN], index: 0, greater: 0, lesser: 0, last: 0 } }

    pub fn push(&mut self, value: usize) -> TrendDirection {
        self.data[self.index] = value;
        let ret = if self.index == 0 {
            self.last = value;
            TrendDirection::None
        } else {
            if value > self.last {
                self.greater += 1;
                TrendDirection::Positive
            } else if value < self.last {
                self.lesser += 1;
                TrendDirection::Negative
            } else {
                TrendDirection::None
            }
        };
        self.index += 1;
        assert!(self.index < TREND_LEN, "Increase FUZZY_LEN");
        ret
    }

    /// Returns positive, zero, or negative value based on the trend of the values recorded
    pub fn trend(&self) -> TrendDirection { (self.greater - self.lesser).into() }

    /// Size of the trend. The sign does not matter.
    pub fn magnitude(&self) -> usize { (self.greater - self.lesser).abs() as usize }
}
/// (0, 0) is at the lower left corner
pub struct ImageLuma<'a> {
    data: &'a mut [u8],
    width: usize,
    height: usize,
    thresh: u8,
    // coordinates of a subimage, if set. The ROI includes these points.
    x0: usize,
    x1: usize,
    y0: usize,
    y1: usize,
    iter_row: RefCell<usize>,
}
impl<'a> ImageLuma<'a> {
    pub fn new(data: &'a mut [u8], dimensions: (u32, u32), thresh: u8) -> Self {
        let (w, h) = dimensions;
        // ROI is default the entire area
        Self {
            data,
            width: w as usize,
            height: h as usize,
            thresh,
            x0: 0,
            x1: w as usize,
            y0: 0,
            y1: h as usize,
            iter_row: RefCell::new(0),
        }
    }

    pub fn dimensions(&self) -> (usize, usize) { (self.width, self.height) }

    pub fn set_thresh(&mut self, t: u8) { self.thresh = t }

    pub fn binarize(&self, luma: u8) -> Color { if luma > self.thresh { Color::White } else { Color::Black } }

    pub fn get_pixel(&self, x: usize, y: usize) -> u8 { self.data[x + y * self.width] }

    pub fn put_pixel(&mut self, x: usize, y: usize, luma: u8) {
        self.data[x + y * self.width as usize] = luma;
    }

    pub fn is_four_connected(&self, p: Point) -> bool {
        if p.x < 1 || p.x >= self.width as isize - 2 || p.y > 1 || p.y >= self.height as isize - 2 {
            false
        } else {
            let xu = p.x as usize;
            let yu = p.y as usize;
            let color = self.get_pixel(xu, yu) <= self.thresh;
            !((self.get_pixel(xu - 1, yu) <= self.thresh) ^ color
                || (self.get_pixel(xu, yu - 1) <= self.thresh) ^ color
                || (self.get_pixel(xu + 1, yu) <= self.thresh) ^ color
                || (self.get_pixel(xu, yu + 1) <= self.thresh) ^ color)
        }
    }

    pub fn set_roi(&mut self, tl: Point, br: Point) {
        assert!(tl.x >= 0);
        assert!(tl.y >= 0);
        assert!(br.x >= 0);
        assert!(br.y >= 0);
        self.x0 = tl.x as usize;
        self.x1 = br.x as usize;
        self.y0 = br.y as usize;
        self.y1 = tl.y as usize;
        *self.iter_row.borrow_mut() = 0;
    }

    pub fn reset_roi(&self) { *self.iter_row.borrow_mut() = 0; }

    pub fn next_roi_row(&'a self) -> Option<(usize, &'a [u8])> {
        let cur_row = *self.iter_row.borrow();
        if cur_row <= self.y1 - self.y0 {
            *self.iter_row.borrow_mut() = cur_row + 1;
            Some((
                cur_row,
                &self.data
                    [(self.y0 + cur_row) * self.width + self.x0..(self.y0 + cur_row) * self.width + self.x1],
            ))
        } else {
            None
        }
    }

    pub fn get_roi_row(&'a self, roi_row: usize) -> Option<&'a [u8]> {
        if roi_row <= self.y1 - self.y0 {
            Some(
                &self.data
                    [(self.y0 + roi_row) * self.width + self.x0..(self.y0 + roi_row) * self.width + self.x1],
            )
        } else {
            None
        }
    }

    pub fn roi_width(&self) -> usize { self.x1 - self.x0 }

    pub fn roi_height(&self) -> usize { self.y1 - self.y0 }

    /// Return 1 if the pixel at x, y matches color; otherwise, 0
    fn to_count(&self, x: isize, y: isize, color: Color) -> usize {
        let index = (y * (self.width as isize) + x).max(0) as usize;
        /*
        println!(
            "x: {} y: {}, result: {}, len: {}, pix {:?}, color: {:?}, result: {:?}",
            x,
            y,
            index,
            self.data.len(),
            self.binarize(self.data[index]),
            color,
            !(self.binarize(self.data[index]) ^ color)
        ); */
        (!(self.binarize(self.data[index]) ^ color)).into()
    }

    pub fn neighbor_count(&self, point: Point, color: Color) -> Option<usize> {
        let x = point.x;
        let y = point.y;
        if x > 0 && x < (self.width - 1) as isize && y > 0 && y < (self.height - 1) as isize {
            // Count 8 neighbors
            let mut count: usize = 0;
            for direction in SearchDirection::iter() {
                let p: Point = direction.into();
                count += self.to_count(x as isize + p.x, y as isize + p.y, color);
            }
            Some(count)
        } else {
            None
        }
    }

    /// This transforms ROI coordinates into image coordinates
    pub fn neighbor_count_roi(&self, point: Point, color: Color) -> Option<usize> {
        self.neighbor_count(self.roi_to_absolute(point).unwrap(), color)
    }

    pub fn roi_to_absolute(&self, point: Point) -> Option<Point> {
        let x = point.x + self.x0 as isize;
        let y = point.y + self.y0 as isize;
        if x >= 0 && x < self.width as isize && y >= 0 && y < self.height as isize {
            Some(Point::new(x, y))
        } else {
            None
        }
    }

    pub fn absolute_to_roi(&self, point: Point) -> Option<Point> {
        let x = point.x - self.x0 as isize;
        let y = point.y - self.y0 as isize;
        if x >= 0 && x < self.x1 as isize && y >= 0 && y < self.y1 as isize {
            Some(Point::new(x, y))
        } else {
            None
        }
    }

    /// Find the corner of a finder.
    pub fn corner_finder(
        // must have an ROI set
        &mut self,
        // this point needs to be transformed to the center of the ROI
        center: Point,
        // putative width of the finder
        finder_width: usize,
        // the expected direction of the corner
        corner_dir: SearchDirection,
        // returns the Point in absolute image offset
    ) -> Option<Point> {
        // constrain our search area to a smaller ROI.
        let search_dir: Point = corner_dir.into();
        // check that we were given a *corner*, not a cardinal
        assert!(search_dir.x != 0);
        assert!(search_dir.y != 0);

        let tl = Point::new(
            (center.x - (finder_width / 2 + finder_width / 7) as isize).max(0),
            (center.y + (finder_width / 2 + finder_width / 7) as isize).min(self.height as isize),
        );
        let br = Point::new(
            (center.x + (finder_width / 2 + finder_width / 7) as isize).min(self.width as isize),
            (center.y - (finder_width / 2 + finder_width / 7) as isize).max(0),
        );
        self.set_roi(tl, br);

        // figure out which edge of the ROI we're starting the search from;
        // set the starting point to be one pixel beyond the edge.
        let start_x = if search_dir.x < 0 { 1 } else { self.roi_width() - 1 };

        // set candidate at the extreme end of the search range (opposite corner of search direction)
        let mut candidate_roi = Point::new(
            if search_dir.x < 0 { self.roi_width() as isize - 1 } else { 1 },
            // if search_dir.y < 0 { 1 } else { self.roi_height() as isize - 1 },
            0,
        );

        // threshold for determining that a pixel should be counted
        const EDGE_THRESH: usize = 3;

        // search past the inflection point by the estimated width of a QR row
        let symbol_width = finder_width / (1 + 1 + 3 + 1 + 1).max(1);
        let mut trend = Trend::new();
        let mut last_dir = TrendDirection::None;
        let mut significance: usize = 0;
        let mut early_quit = false;
        while let Some((y, row)) = self.next_roi_row() {
            for i in 1..row.len() - 1 {
                let x = (start_x as isize + i as isize * -search_dir.x) as usize;
                if self.binarize(row[x]) == Color::Black {
                    if self.neighbor_count_roi(Point::new(x as isize, y as isize), Color::Black).unwrap()
                        >= EDGE_THRESH
                    {
                        let dir = trend.push(x);
                        if dir != last_dir {
                            if candidate_roi.x * search_dir.x < i as isize * search_dir.x {
                                candidate_roi.x = i as isize;
                            }
                            if candidate_roi.y * search_dir.y < y as isize * search_dir.y {
                                candidate_roi.y = y as isize;
                            }
                            last_dir = dir;
                            // declining trend
                            if trend.magnitude() + symbol_width < significance {
                                early_quit = true;
                            }
                        }
                        if trend.magnitude() > significance {
                            significance = trend.magnitude();
                        }
                        break;
                    }
                }
            }
            if early_quit {
                break;
            }
        }

        // If the search didn't terminate at an edge of the ROI, the candidate is the corner point.
        if candidate_roi.x != 0
            && candidate_roi.x != self.roi_width() as isize
            && candidate_roi.y != 0
            && candidate_roi.y != self.roi_height() as isize
        {
            self.roi_to_absolute(candidate_roi)
        } else {
            None
        }
    }
}

pub struct SeqBuffer {
    buffer: [Option<FinderSeq>; 5],
    start: usize,
    end: usize,
    count: usize,
}

impl SeqBuffer {
    pub const fn new() -> Self { SeqBuffer { buffer: [None; SEQ_LEN], start: 0, end: 0, count: 0 } }

    pub fn clear(&mut self) {
        self.buffer = [None; SEQ_LEN];
        self.start = 0;
        self.end = 0;
        self.count = 0;
    }

    pub fn push(&mut self, item: FinderSeq) {
        self.buffer[self.end] = Some(item);
        self.end = (self.end + 1) % SEQ_LEN;

        if self.count < SEQ_LEN {
            self.count += 1;
        } else {
            self.start = (self.start + 1) % SEQ_LEN; // Overwrite the oldest element
        }
    }

    /// Don't use options because we iterate through the list once to extract the
    /// correct ordering, and we know how many valid items are in there. This is less
    /// idiomatic, but it saves us the computational overhead of constantly iterating
    /// through to test for None when we know how many there are in the first place,
    /// and it saves the lexical verbosity of `unwrap()` everywhere (and unwrap does
    /// actually have a computational cost, it is not a zero-cost abstraction).
    pub fn retrieve(&self, output: &mut [FinderSeq; SEQ_LEN]) -> usize {
        let mut copied_count = 0;

        for i in 0..self.count {
            let index = (self.start + i) % SEQ_LEN;
            output[i] = self.buffer[index].expect("circular buffer logic error").clone();
            copied_count += 1;
        }

        for i in copied_count..5 {
            output[i] = FinderSeq::default(); // Clear the remaining elements in the output if any
        }

        copied_count
    }

    // returns a tuple of (center point of the sequence, total length of sequence)
    pub fn search(&self) -> Option<(usize, usize)> {
        let mut ratios = [0usize; SEQ_LEN];
        let mut seq: [FinderSeq; SEQ_LEN] = [FinderSeq::default(); SEQ_LEN];
        if self.retrieve(&mut seq) == SEQ_LEN {
            // only look for sequences that start with black
            if seq[0].color == Color::Black {
                let denom = seq[0].run;
                ratios[0] = 1 << SEQ_FP_SHIFT; // by definition
                for (ratio, s) in ratios[1..].iter_mut().zip(seq[1..].iter()) {
                    *ratio = (s.run << SEQ_FP_SHIFT) / denom;
                }
                if ratios[1] >= LOWER_1
                    && ratios[1] <= UPPER_1
                    && ratios[2] >= LOWER_3
                    && ratios[2] <= UPPER_3
                    && ratios[3] >= LOWER_1
                    && ratios[3] <= UPPER_1
                    && ratios[4] >= LOWER_1
                    && ratios[4] <= UPPER_1
                {
                    // crate::println!("  seq {:?}", &seq);
                    // crate::println!("  ratios {:?}", &ratios);
                    Some((seq[2].pos - seq[2].run / 2 - 1, seq.iter().map(|s| s.run).sum()))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }
}

const ROW_LIMIT: usize = 128;
/// Returns the average width of the finder regions found.
pub fn find_finders(candidates: &mut [Option<Point>], image: &[u8], thresh: u8, stride: usize) -> usize {
    let mut row_candidates: [Option<Point>; ROW_LIMIT] = [None; ROW_LIMIT];

    // ideally, the candidates would be a Vec, but we want this to work without allocations
    // so we're going to insert them manually into a list.
    let mut row_candidate_index = 0;

    let mut seq_buffer = SeqBuffer::new();

    // crate::println!("ROWSROWSROWSROWS");
    for (y, line) in image.chunks(stride).enumerate() {
        seq_buffer.clear();
        let mut last_color = Color::from(line[0], thresh);
        let mut run_length = 1;

        for (x_minus1, &pix) in line[1..].iter().enumerate() {
            let pix = Color::from(pix, thresh);
            if pix == last_color {
                run_length += 1;
            } else {
                seq_buffer.push(FinderSeq { run: run_length, pos: x_minus1 + 1, color: last_color });
                last_color = pix;
                run_length = 1;

                // manage the sequence index
                if let Some((pos, _width)) = seq_buffer.search() {
                    // crate::println!("row candidate {}, {}", pos, y);
                    row_candidates[row_candidate_index] = Some(Point::new(pos as _, y as _));
                    row_candidate_index += 1;
                    if row_candidate_index == row_candidates.len() {
                        // just abort the search if we run out of space to store results
                        break;
                    }
                }
            }
        }
        if row_candidate_index == row_candidates.len() {
            println!("ran out of space for row candidates");
            break;
        }
    }

    // crate::println!("CCCCCCCCCCCCCCCC");
    let mut candidate_index = 0;
    let mut candidate_width = 0;
    for x in 0..stride {
        seq_buffer.clear();

        let mut last_color = Color::from(image[x], thresh);
        let mut run_length = 1;
        // could rewrite this to abort the search after more than 3 finders are found, but for now,
        // do an exhaustive search because it's useful info for debugging.
        for (y_minus1, row) in image[x + stride..].chunks(stride).enumerate() {
            let pix = Color::from(row[0], thresh);
            if pix == last_color {
                run_length += 1;
            } else {
                seq_buffer.push(FinderSeq { run: run_length, pos: y_minus1 + 1, color: last_color });
                last_color = pix;
                run_length = 1;
                if let Some((pos, width)) = seq_buffer.search() {
                    let search_point = Point::new(x as _, pos as _);
                    // crate::println!("col candidate {}, {}", x, pos);

                    // now cross the candidate against row candidates; only report those that match
                    for &rc in row_candidates
                        .iter()
                        .filter(|&&x| if let Some(p) = x { p == search_point } else { false })
                    {
                        if candidate_index < candidates.len() {
                            candidates[candidate_index] = rc;
                            candidate_index += 1;
                            candidate_width += width;
                        } else {
                            // just abort the search if we run out of space to store results
                            break;
                        }
                    }
                }
            }
            if candidate_index == candidates.len() {
                // just abort the search if we run out of space to store results
                println!("ran out of space for processed candidates");
                break;
            }
        }
    }
    candidate_width / candidate_index
}

/// Function to estimate the fourth point from three points. The result is not exact
/// because we don't know the obliqueness of the camera plane with respect to the image.
pub fn estimate_fourth_point(points: &[Point; 3]) -> Point {
    let x4 = points[0].x + points[2].x - points[1].x;
    let y4 = points[0].y + points[2].y - points[1].y;

    Point::new(x4, y4)
}

/// Performs Zhang-Suen Thinning on a binary image represented as a slice of u8.
/// The input image is modified in-place.
///
/// # Arguments
/// - `image`: A mutable slice of `u8` representing the binary image (0 or 1).
/// - `width`: The width of the image.
/// - `height`: The height of the image.
/// - `threshold`: Threshold value to binarize the image. Pixels >= threshold are set to 1, others to 0.
pub fn zhang_suen_thinning(image: &mut [u8], width: usize, height: usize, threshold: u8) {
    // Binarize the image using the given threshold
    binarize_image(image, threshold);

    // Prepare to iterate until no changes are made
    let mut has_changes = true;

    while has_changes {
        has_changes = false;

        // First Sub-iteration
        let mut markers = vec![0u8; image.len()]; // 0: keep, 1: delete
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;
                if image[idx] == 1 && should_remove_pixel(image, x, y, width, true) {
                    markers[idx] = 1;
                    has_changes = true;
                }
            }
        }

        let mut debug = markers.clone();
        unbinarize_image(&mut debug);
        show_image(&DynamicImage::ImageLuma8(GrayImage::from_vec(width as _, height as _, debug).unwrap()));

        // Remove marked pixels
        for (i, &marker) in markers.iter().enumerate() {
            if marker == 1 {
                image[i] = 0;
            }
        }

        // Second Sub-iteration
        let mut markers = vec![0u8; image.len()];
        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let idx = y * width + x;
                if image[idx] == 1 && should_remove_pixel(image, x, y, width, false) {
                    markers[idx] = 1;
                    has_changes = true;
                }
            }
        }

        // Remove marked pixels
        for (i, &marker) in markers.iter().enumerate() {
            if marker == 1 {
                image[i] = 0;
            }
        }
    }
}

/// Binarizes the image based on a threshold.
fn binarize_image(image: &mut [u8], threshold: u8) {
    for pixel in image.iter_mut() {
        *pixel = if *pixel >= threshold { 1 } else { 0 };
    }
}

pub fn unbinarize_image(image: &mut [u8]) {
    for pixel in image.iter_mut() {
        *pixel = if *pixel != 0 { 255 } else { 0 };
    }
}

/// Checks if a pixel should be removed according to the Zhang-Suen criteria.
/// `is_first_iteration` indicates whether it's the first or second sub-iteration.
fn should_remove_pixel(image: &[u8], x: usize, y: usize, width: usize, is_first_iteration: bool) -> bool {
    let idx = y * width + x;

    // Get 8 neighbors
    let p2 = image[(y - 1) * width + x]; // north
    let p3 = image[(y - 1) * width + (x + 1)]; // northeast
    let p4 = image[y * width + (x + 1)]; // east
    let p5 = image[(y + 1) * width + (x + 1)]; // southeast
    let p6 = image[(y + 1) * width + x]; // south
    let p7 = image[(y + 1) * width + (x - 1)]; // southwest
    let p8 = image[y * width + (x - 1)]; // west
    let p9 = image[(y - 1) * width + (x - 1)]; // northwest

    // Compute the number of non-zero neighbors
    let non_zero_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

    // Count the number of 0 to 1 transitions in the sequence P2 -> P3 -> P4 -> P5 -> P6 -> P7 -> P8 -> P9 ->
    // P2
    let transitions = count_transitions(&[p2, p3, p4, p5, p6, p7, p8, p9, p2]);

    // Conditions
    if non_zero_count >= 2
        && non_zero_count <= 6
        && transitions == 1
        && (is_first_iteration && (p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0))
        || (!is_first_iteration && (p2 * p4 * p8 == 0) && (p2 * p6 * p8 == 0))
    {
        return true;
    }

    false
}

/// Counts the number of 0 to 1 transitions in the given neighborhood sequence.
fn count_transitions(neighbors: &[u8]) -> u8 {
    let mut count = 0;
    for i in 0..neighbors.len() - 1 {
        if neighbors[i] == 0 && neighbors[i + 1] == 1 {
            count += 1;
        }
    }
    count
}
