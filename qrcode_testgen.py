import qrcode
import random
import string

# Generate a random string of characters (for example, 500 characters)
random_data = ''.join(random.choices(string.ascii_letters + string.digits, k=500))

# Create a QR code with version 11 and the appropriate error correction level
qr = qrcode.QRCode(
    version=8,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data("TOTP code is about this long in some cases and i have seen even longer, about this long?")
qr.make(fit=True)

# Generate the image
img = qr.make_image(fill_color="black", back_color="white")

# Save the image
img.save('./random_qr_code_version8.png')

print("QR Code generated successfully!")