package xai.hlta

import org.latlab.model.LTM
import org.latlab.util.{DataSet, ScoreCalculator}

trait BICScoreCalculator {
  def compute(model: LTM, data: DataSet): Double =
    compute(model, data, model.computeDimension)

  def compute(model: LTM, data: DataSet, dim: Int): Double =
    compute(computeLL(model, data), dim, data.getTotalWeight)

  def compute(ll: Double, dim: Int, size: Double): Double =
    ll - dim * Math.log(size) / 2.0

  def computeLL(model: LTM, data: DataSet) =
    ScoreCalculator.computeLoglikelihood(model, data)
}

object BICScoreCalculator {
  def create(batchSize: Int, dataSize: Double) = {
    if (batchSize > 0 && batchSize < dataSize) new BatchBICScoreCalculator(batchSize)
    else NotBatchBICScoreCalculator
  }
}

object NotBatchBICScoreCalculator extends BICScoreCalculator

class BatchBICScoreCalculator(val batchSize: Int) extends BICScoreCalculator {
  override def compute(model: LTM, data: DataSet, dim: Int) = {
    val size = data.getTotalWeight

    if (size < batchSize)
      super.compute(model, data, dim)
    else {
      val numberOfBatches = size / batchSize
      compute(computeLL(model, data) / numberOfBatches, dim, batchSize)
    }
  }
}
