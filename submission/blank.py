"""D11 inference-only submission. Training is performed locally."""


class Defense:
    """Expose the fitted estimator attributes restored by the D11 evaluator."""

    name = "face_region"
    feature_kind = "face_region"
    feature_dim = 61

    def __init__(self, C=1.0, max_iter=500, seed=42, **kwargs):
        # All imports stay inside the class for platforms that transplant its AST.
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.C, self.max_iter, self.seed = C, max_iter, seed
        self._clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed, class_weight="balanced")
        self._scaler = StandardScaler()
        self._fitted = False

    def _augment(self, image, kind, rng):
        import io
        import numpy as np
        from PIL import Image, ImageFilter

        image = np.asarray(image, dtype=np.uint8)
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        if kind == "jpeg":
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            with Image.open(buf) as decoded:
                return np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        if kind == "resize_recover":
            size = int(rng.choice((192, 224)))
            filters = [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC,
                       Image.Resampling.BILINEAR, Image.Resampling.LANCZOS]
            small = pil.resize((size, size), filters[int(rng.randint(4))])
            return np.asarray(small.resize((w, h), filters[int(rng.randint(4))]), dtype=np.uint8)
        if kind == "blur_sharpen":
            radius = 1.2 * (0.8 + 0.4 * float(rng.rand()))
            result = pil.filter(ImageFilter.GaussianBlur(radius)).filter(
                ImageFilter.UnsharpMask(radius=max(1, int(round(radius))), percent=150, threshold=0)
            )
            return np.asarray(result, dtype=np.uint8)
        if kind == "color_laundering":
            img = image.astype(np.float32) / 255.0
            gamma = 0.9 * (0.95 + 0.1 * float(rng.rand()))
            saturation = 1.1 * (0.9 + 0.2 * float(rng.rand()))
            temperature = float(rng.uniform(-0.03, 0.03))
            img = np.clip(img, 0, 1) ** (1.0 / gamma)
            gray = img.mean(axis=2, keepdims=True)
            img = gray + (img - gray) * saturation
            img[..., 0] = np.clip(img[..., 0] + temperature, 0, 1)
            img[..., 2] = np.clip(img[..., 2] - temperature, 0, 1)
            return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        if kind == "frequency_laundering":
            img = image.astype(np.float32)
            yy, xx = np.mgrid[:h, :w]
            radius = np.sqrt(((yy - h // 2) / (h // 2)) ** 2 + ((xx - w // 2) / (w // 2)) ** 2)
            mask = 1.0 / (1.0 + np.exp((radius - 0.35) * 12.0))
            result = np.empty_like(img)
            for ch in range(3):
                spectrum = np.fft.fftshift(np.fft.fft2(img[..., ch]))
                low = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum * mask)))
                result[..., ch] = img[..., ch] * 0.4 + low * 0.6
            return np.clip(result, 0, 255).astype(np.uint8)
        if kind == "composite":
            for step in ("jpeg", "resize_recover", "blur_sharpen"):
                image = self._augment(image, step, rng)
            return image
        raise ValueError("Unknown augmentation")

    def fit(self, images, labels):
        """Local training entry point used by official local_train_media.py only."""
        import numpy as np

        images = np.asarray(images)
        labels = np.asarray(labels, dtype=np.int64)
        if len(images) != len(labels) or set(np.unique(labels)) != {0, 1}:
            raise ValueError("Training requires aligned images and both binary classes")
        self._fitted = False
        features = [self._extract(image) for image in images]
        targets = labels.tolist()
        for i, kind in enumerate(("jpeg", "resize_recover", "blur_sharpen",
                                  "color_laundering", "frequency_laundering", "composite")):
            rng = np.random.RandomState(self.seed + i)
            for image in images[labels == 1]:
                features.append(self._extract(self._augment(image, kind, rng)))
                targets.append(1)
        X = np.stack(features)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite training features")
        self._clf.fit(self._scaler.fit_transform(X), targets)
        self._fitted = True
        return self

    def _extract(self, image):
        import numpy as np

        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3 or min(img.shape[:2]) < 32:
            raise ValueError("Expected an RGB image of at least 32 x 32 pixels")
        if not np.isfinite(img).all():
            raise ValueError("Image contains non-finite values")
        img = img.astype(np.float32)
        h, w, _ = img.shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        oval = ((xx - w / 2.0) / (w * 0.32)) ** 2
        oval += ((yy - h / 2.0) / (h * 0.40)) ** 2
        masks = [
            oval < 0.55,
            (oval >= 0.55) & (oval < 1.00),
            (oval >= 0.85) & (oval < 1.15),
            (oval >= 1.15) & (oval < 1.70),
            yy < h * 0.16,
        ]
        feats = []
        means = []
        gray = np.mean(img, axis=2)
        for mask in masks:
            vals = img[mask]
            means.append(vals.mean(axis=0))
            feats.extend(float(v) for v in vals.mean(axis=0))
            feats.extend(float(v) for v in vals.std(axis=0))
            feats.extend(float(v) for v in (
                np.percentile(vals, 95, axis=0) - np.percentile(vals, 5, axis=0)
            ))
        for a, b in ((0, 3), (1, 3), (0, 1), (2, 3)):
            feats.extend(float(v) for v in means[a] - means[b])
        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        feats.extend([
            float(np.mean(np.abs(gx[:, int(w * 0.47):int(w * 0.53)]))),
            float(np.mean(np.abs(gy[int(h * 0.35):int(h * 0.66), :]))),
            float(np.std(gray[masks[2]])),
            float(np.std(gray[masks[0]])),
        ])
        return np.asarray(feats, dtype=np.float32)

    def predict(self, image):
        import numpy as np
        from sklearn.utils.validation import check_is_fitted

        # The evaluator restores these two attributes directly; no fit flag needed.
        check_is_fitted(self._clf)
        check_is_fitted(self._scaler)
        features = self._extract(image).reshape(1, -1)
        decision = float(self._clf.decision_function(
            self._scaler.transform(features)
        )[0]) + 0.75
        return float(np.exp(-np.logaddexp(0.0, -decision)))

    def predict_batch(self, images):
        return [self.predict(image) for image in images]

    def defend(self, request):
        return {"fake_probability": self.predict(request["image"])}

    def load_model(self, model_path):
        """Optional local convenience; the platform may restore attributes itself."""
        import io
        import joblib
        import torch
        from sklearn.utils.validation import check_is_fitted

        container = torch.load(model_path, map_location="cpu", weights_only=True)
        if container.get("format") != "joblib-v1":
            raise ValueError("Expected the official joblib-v1 PyTorch container")
        payload = container["payload"]
        if payload.dtype != torch.uint8 or payload.ndim != 1:
            raise ValueError("Checkpoint payload must be a flat uint8 tensor")
        state = joblib.load(io.BytesIO(payload.numpy().tobytes()))
        if state["feature_kind"] != self.feature_kind:
            raise ValueError("Checkpoint uses a different feature extractor")
        if state["feature_dim"] != self.feature_dim:
            raise ValueError("Checkpoint feature dimension mismatch")
        clf, scaler = state["clf"], state["scaler"]
        check_is_fitted(clf)
        check_is_fitted(scaler)
        if clf.n_features_in_ != self.feature_dim or scaler.n_features_in_ != self.feature_dim:
            raise ValueError("Estimator feature dimension mismatch")
        self._clf, self._scaler = clf, scaler
        self._fitted = True
        return self
