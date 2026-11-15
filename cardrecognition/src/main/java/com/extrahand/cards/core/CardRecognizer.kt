package com.extrahand.cards.core

/**
 * THE snap-in interface. Everything a consuming app needs: hand it a cropped card image, get a guess.
 *
 * Get an instance from `CardRecognition.load(...)` (Android, com.extrahand.cards.android) or build one
 * directly with [KnnCardRecognizer.fromDataset] on any JVM.
 */
interface CardRecognizer {
    /** Which feature extractor this recognizer uses; vectors from other extractors are incompatible. */
    val extractorId: String

    /** Number of labelled samples the recognizer knows. 0 = it will always return [Recognition.NONE]. */
    val sampleCount: Int

    /** Samples per card label, e.g. {"AS": 12, "10H": 9}. */
    fun labelCounts(): Map<String, Int>

    /**
     * Recognize a card. [image] should contain one upright-or-upside-down card roughly filling the frame
     * (the guide-box crop from the camera is fine; it is tightened with [CardLocator] unless [locate] is false).
     */
    fun recognize(image: CardImage, topN: Int = 3, locate: Boolean = true): Recognition

    /** Recognize from a precomputed feature vector (must come from the same extractor). */
    fun recognizeFeatures(features: FloatArray, topN: Int = 3): Recognition
}

/** A recognizer that can learn from labelled examples at runtime (used by the trainer app). */
interface TrainableCardRecognizer : CardRecognizer {
    val extractor: FeatureExtractor
    fun learn(sampleId: String, card: Card, features: FloatArray)
    fun forget(sampleId: String): Boolean
    fun clear()
}

data class Candidate(
    val card: Card,
    /** Soft-vote probability among the k nearest neighbours, 0..1. */
    val probability: Float,
    /** Best cosine similarity of any sample with this label, -1..1 (typically 0.5..1). */
    val similarity: Float,
)

data class Recognition(
    /** Best guess, or null if the recognizer has no data or the result is below the acceptance thresholds. */
    val card: Card?,
    /** Probability of [card] (0 when null). */
    val confidence: Float,
    /** Cosine similarity of the closest sample (0 when no data). */
    val similarity: Float,
    /** Top-N candidates (may be non-empty even when [card] is null, i.e. low confidence). */
    val candidates: List<Candidate>,
    /** The feature vector of the query, so callers can learn from it without recomputing. */
    val features: FloatArray?,
    /** The (possibly tightened) card crop that was analysed. */
    val croppedImage: CardImage?,
) {
    val isConfident: Boolean get() = card != null

    companion object {
        val NONE = Recognition(null, 0f, 0f, emptyList(), null, null)
    }
}

/** Acceptance thresholds for reporting [Recognition.card]. */
data class RecognitionThresholds(val minProbability: Float = 0.5f, val minSimilarity: Float = 0.75f)

/** Default implementation: [FeatureExtractor] + [KnnIndex]. Thread-safe. */
class KnnCardRecognizer(
    override val extractor: FeatureExtractor = LumaGridV1,
    private val thresholds: RecognitionThresholds = RecognitionThresholds(),
    private val k: Int = 7,
) : TrainableCardRecognizer {
    private val index = KnnIndex(extractor.dim)

    override val extractorId: String get() = extractor.id
    override val sampleCount: Int get() = index.size
    override fun labelCounts(): Map<String, Int> = index.labelCounts()

    override fun recognize(image: CardImage, topN: Int, locate: Boolean): Recognition {
        val crop = if (locate) CardLocator.locate(image).image else image
        val f = extractor.extract(crop)
        return recognizeFeatures(f, topN).copy(croppedImage = crop)
    }

    override fun recognizeFeatures(features: FloatArray, topN: Int): Recognition {
        val scores = index.classify(features, k)
        if (scores.isEmpty()) return Recognition.NONE.copy(features = features)
        val candidates = scores.take(topN).mapNotNull { s ->
            Card.parse(s.label)?.let { Candidate(it, s.probability, s.similarity) }
        }
        val best = candidates.firstOrNull()
        val accept = best != null && best.probability >= thresholds.minProbability &&
            best.similarity >= thresholds.minSimilarity
        return Recognition(
            card = if (accept) best!!.card else null,
            confidence = if (accept) best!!.probability else 0f,
            similarity = scores.maxOf { it.similarity },
            candidates = candidates,
            features = features,
            croppedImage = null,
        )
    }

    override fun learn(sampleId: String, card: Card, features: FloatArray) = index.add(sampleId, card.label, features)
    override fun forget(sampleId: String): Boolean = index.remove(sampleId)
    override fun clear() = index.clear()

    /** Add every sample of a decoded dataset. Samples whose label does not parse are skipped. */
    fun addAll(samples: List<SampleRecord>) {
        for (s in samples) {
            val card = Card.parse(s.label) ?: continue
            index.add(s.id, card.label, DatasetCodec.decodeVector(s.f, extractor.dim))
        }
    }

    companion object {
        /**
         * Build a recognizer from a manifest and its shards (already downloaded/parsed).
         * Throws if the dataset was produced by a different extractor.
         */
        fun fromDataset(
            manifest: DatasetManifest,
            shards: List<FeatureShard>,
            extractor: FeatureExtractor = LumaGridV1,
            thresholds: RecognitionThresholds = RecognitionThresholds(),
        ): KnnCardRecognizer {
            require(manifest.extractor == extractor.id) {
                "Dataset extractor '${manifest.extractor}' != '${extractor.id}'. Rebuild features or use a matching extractor."
            }
            return KnnCardRecognizer(extractor, thresholds).apply { addAll(DatasetCodec.liveSamples(manifest, shards)) }
        }
    }
}
