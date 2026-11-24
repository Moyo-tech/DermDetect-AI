"""Post-install checks on invented JPEGs only; no dataset inputs."""
import io
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dermdetect.data.image_integrity import decode_fingerprint
from dermdetect.data.schemas import ContractError


def verify():
    from PIL import Image
    def jpeg(mode="RGB", size=(19, 17), exif=None):
        with Image.new(mode, size, 100) as image:
            stream = io.BytesIO()
            options = {"exif": exif} if exif is not None else {}
            image.save(stream, format="JPEG", **options)
            return stream.getvalue()
    normal = jpeg()
    decoded = decode_fingerprint(normal)
    assert (decoded["width"], decoded["height"]) == (19, 17)
    assert decoded == decode_fingerprint(normal)
    assert decode_fingerprint(jpeg("L"))["dhash64"] == 0
    exif = Image.Exif()
    exif[274] = 6
    rotated = decode_fingerprint(jpeg(exif=exif))
    assert (rotated["width"], rotated["height"]) == (17, 19)
    for payload in (b"INVENTED TEXT; NOT AN IMAGE", normal[:len(normal) // 2], normal[:-2]):
        try:
            decode_fingerprint(payload)
        except ContractError:
            pass
        else:
            raise AssertionError("Invalid/truncated JPEG accepted")
    try:
        decode_fingerprint(jpeg("CMYK"))
    except ContractError as exc:
        assert exc.code == "unsupported_image"
    else:
        raise AssertionError("Unsupported mode accepted")
    print("Invented JPEG checks passed: full load, determinism, luminance, EXIF, text/truncation and mode refusal")


if __name__ == "__main__":
    verify()
