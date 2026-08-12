const LOW_CONFIDENCE_THRESHOLD = 0.6;
const CLOSE_RUNNER_UP_GAP = 0.15;

export default function ResultCard({ result }) {
  const sortedProbabilities = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]);
  const [, runnerUpProbability] = sortedProbabilities[1] ?? [null, 0];
  const isLowConfidence = result.confidence < LOW_CONFIDENCE_THRESHOLD;
  const isCloseCall = result.confidence - runnerUpProbability < CLOSE_RUNNER_UP_GAP;

  return (
    <div className="result-card">
      <div className="result-headline">
        <span className="result-label">Predicted condition</span>
        <span className="result-class">{result.predicted_class}</span>
        <span className="result-confidence">{(result.confidence * 100).toFixed(1)}% confidence</span>
      </div>

      {(isLowConfidence || isCloseCall) && (
        <p className="uncertainty-banner">
          {isLowConfidence
            ? "Low confidence — this model was trained on under 900 images, so treat this result as a hint, not a verdict."
            : `Close call — "${sortedProbabilities[1][0]}" is also plausible. Consider a clearer or closer photo.`}
        </p>
      )}

      <div className="probability-list">
        {sortedProbabilities.map(([className, probability]) => (
          <div className="probability-row" key={className}>
            <span className="probability-name">{className}</span>
            <div className="probability-track">
              <div
                className="probability-fill"
                style={{ width: `${Math.max(probability * 100, 2)}%` }}
              />
            </div>
            <span className="probability-value">{(probability * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
