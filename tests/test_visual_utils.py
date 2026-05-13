import pytest

from scripts.main_pipeline import require_real_data_file
from scripts.route_visualization import _estimate_deicing_kg, bearing_label
from scripts.shadow_utils import solar_position_kst


def test_bearing_label_cardinal_directions():
    assert bearing_label((0, 0), (0, 1)) == "북"
    assert bearing_label((0, 0), (1, 0)) == "동"
    assert bearing_label((0, 0), (0, -1)) == "남"
    assert bearing_label((0, 0), (-1, 0)) == "서"


def test_estimate_deicing_kg_uses_pipeline_default_when_rate_missing():
    assert _estimate_deicing_kg({"area": 2000}) == pytest.approx(60)


def test_solar_position_kst_returns_reasonable_winter_noon_elevation():
    elevation, azimuth = solar_position_kst(37.514, 127.047, 1, 15, 12)

    assert 20 <= elevation <= 40
    assert 0 <= azimuth <= 360


def test_require_real_data_file_rejects_git_lfs_pointer(tmp_path):
    pointer = tmp_path / "MOCT_LINK.shp"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\n"
        "size 123\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Git LFS pointer"):
        require_real_data_file(str(pointer), "도로 Shapefile")
