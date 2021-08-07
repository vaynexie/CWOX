package xai.hlta

import org.latlab.learner.ParallelEmLearner
import org.latlab.learner.ParallelStepwiseEmLearner
import org.latlab.learner.SparseDataSet
import org.latlab.model.LTM
import org.latlab.util.DataSet
import org.latlab.model.BeliefNode

import collection.JavaConversions._
import xai.util.Reader
import xai.util.Data

object EM{
  
  private class DataOrDataSet[T]
  private object DataOrDataSet {
    implicit object DataWitness extends DataOrDataSet[Data]
    implicit object DataSetWitness extends DataOrDataSet[DataSet]
  }
  
  def apply[T: DataOrDataSet](model: LTM, data: T, steps: Int = 50, numOfRestarts: Int = 5
      , threshold: Double = 1e-2, smooth: Boolean = true) = 
  data match {
    case _: DataSet => em(model, data.asInstanceOf[DataSet], steps, numOfRestarts, threshold, smooth)
    case _: Data => em(model, data.asInstanceOf[Data].toHlcmDataSet, steps, numOfRestarts, threshold, smooth)
  }
 
  private def em(model: LTM, data: DataSet, steps: Int, numOfRestarts: Int, threshold: Double, smooth: Boolean) = {
    val emLearner = new ParallelEmLearner();
    emLearner.setLocalMaximaEscapeMethod("ChickeringHeckerman");
    emLearner.setMaxNumberOfSteps(steps);
    emLearner.setNumberOfRestarts(numOfRestarts);
    emLearner.setReuseFlag(true);
    emLearner.setThreshold(threshold);

    val modelAfterEM = emLearner.em(model, data).asInstanceOf[LTM]
    if(smooth)
      smoothParameters(modelAfterEM, data.getTotalWeight)
    else
      modelAfterEM
  }
  
  private def reorderStates(model: LTM) : LTM = xai.hlta.EmMethods.reorderStates(model)
  
  private def smoothParameters(model: LTM, sampleSize: Double) = {
    model.getNodes.map(_.asInstanceOf[BeliefNode]).foreach { n =>
      val cardinality = n.getVariable.getCardinality
      val values = n.getCpt.getCells
      (0 until values.size).foreach { i =>
        values(i) = (values(i) * sampleSize + 1) / (sampleSize + cardinality)
      }
    }

    model
  }
}


object RunEM {
  def main(args: Array[String]) {
    
    if (args.length < 2) {
      printUsage()
    } else {
      run(args(0), args(1))
    }
    
    def printUsage() = {
      println("RunEM model_file data_file")
    }
    
    def run(modelFile: String, dataFile: String) = {
      val (model, data) = Reader.readLTMAndHLCM_native(modelFile, dataFile)
      val modelAfterEM = EM(model, data)
      
      val outputName = modelFile.replaceAll(".bif$", ".em.bif")
      modelAfterEM.saveAsBif(outputName)
    }

  }
}