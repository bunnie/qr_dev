/// This function takes a fraction described as `m.n`, where n is a fraction out of 1024, as well
/// as an `end` value, and returns numbers going from `0..end` such that with each
/// call of `next()` you produce a usize index in the range of `0..end` that is the
/// nearest algebraic integer that would result from adding the value `m.n` to the
/// current iteration count.
///
/// Thus, a value of 1.5 would be passed as m=1, n=512 (as 512/1024 = 0.5).
///
/// This "iterator" is not guaranteed to return every element in the range of `0..end`,
/// and in fact some numbers may not be returned at all, including `0` or `end-1`. However,
/// it does guarantee that the output is monatomically increasing, unless an error signal
/// is injected that is big enough to cause the count to go backwards.
///
/// Features that we might need:
///   - A report of the "error sign", expressed as +1, 0, -1, which expresses the direction
///    in which we're rounding. For example, 6.3 would be returned as 6 with an error of +1.
///    whereas 6.9 would be returned as 7 with an error of -1.
///   - The ability to inject an error correction signal into the fractional counter. This
///    if the loop internally concludes that an error signal would produce a more accurate
///    sub-pixel sampling, the error count would be applied at that iteration so that subsequent
///    indices absorb the delta into their future counts.

const FIXED_POINT: usize = 1 << 10; // 1024

pub struct FracInt {
    m0: usize,
    n0: usize,
    m: usize,
    n: usize,
    end: usize,
    finished: bool,
}
impl FracInt {
    pub fn new(m: usize, n: usize, end: usize) -> Self {
        assert!(n < FIXED_POINT);
        Self { m0: m, n0: n, m: 0, n: 0, end, finished: false }
    }

    pub fn next(&mut self) -> Option<usize> {
        // short circuit computation if we've hit the end of the iterator
        if self.finished {
            return None;
        }
        self.m += self.m0;
        self.n += self.n0;
        // carry the fraction, if needed
        if (self.n >> 10) > 0 {
            self.m += 1;
            self.n &= (1 << 10) - 1;
        }
        // perform rounding on n
        let rounded_m = if self.n >= (FIXED_POINT >> 1) { self.m + 1 } else { self.m };
        if rounded_m < self.end {
            Some(rounded_m)
        } else {
            self.finished = true;
            None
        }
    }

    pub fn reset(&mut self) {
        self.finished = false;
        self.m = 0;
        self.n = 0;
    }

    pub fn error(&self) -> isize {
        if self.n == 0 {
            0
        } else if self.n >= (FIXED_POINT >> 1) {
            1
        } else {
            -1
        }
    }

    /// The sign on m applies to n, but n is represented as a sign-less quantity
    pub fn nudge(&mut self, m: isize, n: usize) {
        assert!(n < FIXED_POINT);
        if m >= 0 {
            self.m = (self.m as isize + m) as usize;
            self.n = self.n + n;

            // carry the fraction, if needed
            if (self.n >> 10) > 0 {
                self.m += 1;
                self.n &= (1 << 10) - 1;
            }
        } else {
            // nudge down case
            self.m = (self.m as isize + m) as usize;
            if self.n >= n {
                // no underflow, just decrement the fraction
                self.n = self.n - n;
            } else {
                self.n = FIXED_POINT - (n - self.n);
                self.m -= 1;
            }
        }
    }
}

/// The official spec term for a pixel of a QR code is a "module"
pub struct ModuleExtract<'a> {
    data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    thresh: u8,
}

// Next up: look at the table of valid codes and figure out the fractional pixel value
// we should use to iterate through the resized QR array.
