package xai.hlta

import org.latlab.learner.ParallelStepwiseEmLearner
import xai.util.{Arguments, Reader}

object StepwiseEm {

  class Conf(args: Seq[String]) extends Arguments(args) {
    banner(
      """Usage: StepwiseEm [OPTION]... sparse_data_file model_file
        |E.g. HLTA data.sparse.txt model1.bif
        |The output file will be model1.estimated.bif""")

    val data = trailArg[String]()
    val model = trailArg[String]()

    val emThreshold = opt[Double](descr = "Threshold of improvement to stop EM (e.g. 0.01) <paper section 6.1>", default = Some(0.01))

    val globalBatchSize = opt[Int](descr = "Number of data cases used in each stepwise EM step. <paper section 7>", default = Some(1000))
    val globalMaxEpochs = opt[Int](descr = "Number of times the whole training dataset has been gone through (e.g. 10). <paper section 7>", default = Some(10))
    val globalMaxEmSteps = opt[Int](descr = "Maximum number of stepwise EM steps (e.g. 128). <paper section 7>", default = Some(128))

    verify
    checkDefaultOpts()
  }

  def main(args: Array[String]): Unit = {
    val conf = new Conf(args)
    println(s"Model: ${conf.model()}")
    println(s"Data: ${conf.data()}")
    val (model, data) = Reader.readLTMAndTuple(conf.model(), conf.data())
    val sparseData = data.toTupleSparseDataSet()

    val batchSize = if (sparseData.getTotalWeight > conf.globalBatchSize())
      conf.globalBatchSize() else sparseData.getTotalWeight
    val threshold = conf.emThreshold()
    val maxSteps = conf.globalMaxEmSteps()
    val maxEpochs = conf.globalMaxEpochs()

    println(s"Batch size: $batchSize, Max steps: $maxSteps, Max epochs: $maxEpochs")

    val estimatedModel = ParallelStepwiseEmLearner.run(sparseData, model,
      1, true, threshold, maxSteps, batchSize, maxEpochs)

    val outputName = conf.model().replaceAll("model\\.bif$", "estimated-model.bif")
    estimatedModel.saveAsBif(outputName)
    print(s"Output file: $outputName")

    // TODO: This may need post-process (to reorder state) and to smooth parameters
  }
}
