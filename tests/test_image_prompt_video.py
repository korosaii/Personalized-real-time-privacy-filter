from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from privacy_filter.image_prompt_video import (
    ReferencePrototype,
    _openvino_reference_fingerprint,
    _reference_prompt_fingerprint,
    build_reference_prototypes,
    encode_reference_prototypes,
    save_mask_image,
)
from privacy_filter.grounded_sam2_video import _union_sam2_logits


class _FakeTorch:
    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def from_numpy(value: np.ndarray) -> np.ndarray:
        return value

    @staticmethod
    def cat(values: list[np.ndarray], dim: int) -> np.ndarray:
        return np.concatenate(values, axis=dim)


class _FakeModel:
    task = "segment"

    def __init__(self) -> None:
        self.model = object()
        self.names: list[str] | None = None
        self.embeddings: np.ndarray | None = None

    def set_classes(self, names: list[str], embeddings: np.ndarray) -> None:
        self.names = names
        self.embeddings = embeddings


class _FakePredictor:
    visual_shapes: list[tuple[int, ...]] = []

    def __init__(self, overrides: dict[str, object]) -> None:
        self.overrides = overrides
        self.prompts: dict[str, np.ndarray] = {}

    def setup_model(self, model: object, verbose: bool) -> None:
        self.model = model

    def set_prompts(self, prompts: dict[str, np.ndarray]) -> None:
        self.prompts = prompts

    def get_vpe(self, image: np.ndarray) -> np.ndarray:
        visuals = self._process_single_image(
            (64, 64),
            image.shape[:2],
            self.prompts["cls"],
            masks=self.prompts["masks"],
        )
        self.visual_shapes.append(visuals.shape)
        return np.ones((1, 1, 4), dtype=np.float32)


class ReferencePrototypeTests(unittest.TestCase):
    def test_visual_prompt_fingerprint_is_deterministic_and_content_sensitive(
        self,
    ) -> None:
        prototype = ReferencePrototype(
            image=np.zeros((8, 12, 3), dtype=np.uint8),
            mask=np.pad(np.ones((4, 6), dtype=bool), ((2, 2), (3, 3))),
            object_index=0,
            path=Path("reference.png"),
            uses_sam_mask=True,
        )
        first = _reference_prompt_fingerprint(
            [prototype], yolo_imgsz=640, yolo_reference_imgsz=640
        )
        second = _reference_prompt_fingerprint(
            [prototype], yolo_imgsz=640, yolo_reference_imgsz=640
        )
        changed = ReferencePrototype(
            image=prototype.image.copy(),
            mask=np.roll(prototype.mask, 1, axis=1),
            object_index=0,
            path=prototype.path,
            uses_sam_mask=True,
        )
        third = _reference_prompt_fingerprint(
            [changed], yolo_imgsz=640, yolo_reference_imgsz=640
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, third)

    def test_openvino_metadata_reads_only_valid_sha256(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            metadata = directory / "metadata.yaml"
            expected = "a" * 64
            metadata.write_text(
                f"task: segment\nvisual_prompt_sha256: {expected}\n",
                encoding="utf-8",
            )
            self.assertEqual(_openvino_reference_fingerprint(directory), expected)
            metadata.write_text(
                "visual_prompt_sha256: invalid\n", encoding="utf-8"
            )
            self.assertIsNone(_openvino_reference_fingerprint(directory))

    def test_sam_video_logits_accept_leading_or_trailing_singleton_channels(self) -> None:
        first = np.zeros((2, 1, 4, 5), dtype=np.float32)
        first[0, 0, 1, 2] = 1.0
        second = np.moveaxis(first, 1, -1)

        expected = np.zeros((4, 5), dtype=bool)
        expected[1, 2] = True
        np.testing.assert_array_equal(_union_sam2_logits(first, 4, 5), expected)
        np.testing.assert_array_equal(_union_sam2_logits(second, 4, 5), expected)

    def test_save_mask_image_writes_viewable_binary_png(self) -> None:
        with TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "nested" / "000000007.png"
            mask = np.array([[False, True], [True, False]], dtype=bool)

            save_mask_image(mask, path)

            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)
            np.testing.assert_array_equal(
                image,
                np.array([[0, 255], [255, 0]], dtype=np.uint8),
            )

    def test_each_photo_becomes_an_independent_exact_mask_prototype(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            first = directory / "front.jpg"
            second = directory / "side.jpg"
            third = directory / "other.jpg"
            image = np.full((100, 120, 3), 127, dtype=np.uint8)
            for path in (first, second, third):
                self.assertTrue(cv2.imwrite(str(path), image))

            first_mask = np.zeros((100, 120), dtype=bool)
            first_mask[20:80, 30:90] = True
            second_mask = np.zeros((100, 120), dtype=bool)
            second_mask[10:90, 20:100] = True
            prototypes = build_reference_prototypes(
                [(first, second), (third,)],
                {first: first_mask, second: second_mask},
                maximum_size=1280,
            )

            self.assertEqual(len(prototypes), 3)
            self.assertEqual([item.object_index for item in prototypes], [0, 0, 1])
            self.assertEqual([item.path for item in prototypes], [first, second, third])
            self.assertTrue(prototypes[0].uses_sam_mask)
            self.assertTrue(prototypes[1].uses_sam_mask)
            self.assertFalse(prototypes[2].uses_sam_mask)
            self.assertEqual(prototypes[0].mask.dtype, np.bool_)
            self.assertLess(prototypes[0].image.shape[0], image.shape[0])
            self.assertLess(prototypes[0].image.shape[1], image.shape[1])
            self.assertEqual(prototypes[0].mask.shape, prototypes[0].image.shape[:2])
            self.assertTrue(np.all(prototypes[2].mask))

    def test_reference_maximum_size_only_downscales(self) -> None:
        with TemporaryDirectory() as directory_name:
            path = Path(directory_name) / "large.png"
            image = np.zeros((200, 400, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            prototype = build_reference_prototypes(
                [(path,)],
                None,
                maximum_size=100,
            )[0]
            self.assertEqual(prototype.image.shape[:2], (50, 100))
            self.assertEqual(prototype.mask.shape, (50, 100))

    def test_encoder_keeps_prototypes_separate_and_returns_object_mapping(self) -> None:
        _FakePredictor.visual_shapes.clear()
        prototypes = [
            ReferencePrototype(
                image=np.zeros((32, 48, 3), dtype=np.uint8),
                mask=np.pad(np.ones((16, 24), dtype=bool), ((8, 8), (12, 12))),
                object_index=0,
                path=Path("front.jpg"),
                uses_sam_mask=True,
            ),
            ReferencePrototype(
                image=np.zeros((48, 32, 3), dtype=np.uint8),
                mask=np.pad(np.ones((24, 16), dtype=bool), ((12, 12), (8, 8))),
                object_index=0,
                path=Path("side.jpg"),
                uses_sam_mask=True,
            ),
        ]
        model = _FakeModel()

        mapping = encode_reference_prototypes(
            model,
            prototypes,
            predictor_type=_FakePredictor,
            device="cpu",
            imgsz=64,
            torch_module=_FakeTorch,
        )

        self.assertEqual(mapping, (0, 0))
        self.assertEqual(model.names, ["prototype0", "prototype1"])
        self.assertIsNotNone(model.embeddings)
        self.assertEqual(model.embeddings.shape, (1, 2, 4))
        self.assertEqual(_FakePredictor.visual_shapes, [(1, 8, 8), (1, 8, 8)])


if __name__ == "__main__":
    unittest.main()
