package org.latlab.learner

import collection.JavaConverters._
import java.util.zip.GZIPInputStream

import xai.test.BaseSpec
import xai.util.Reader
import org.latlab.util.DataSet
import org.latlab.util.DataSet.DataCase
import xai.util.Reader.ARFFToData

import java.util.ArrayList
import java.util.Arrays

class DataOpsSpec extends BaseSpec {

  trait TestData {
    val data: xai.util.Data
    lazy val sparseData = data.toTupleSparseDataSet()
    lazy val wholeData = sparseData.getWholeDenseData
  }

  trait SmallTestData extends TestData {
    val data = Reader.readARFF_native(
      getClass.getResourceAsStream("/sparse-catdata.arff")).toData()
  }

  trait Papers500 extends TestData  {
    val data = Reader.readARFF_native(new GZIPInputStream(
      getClass.getResourceAsStream("/papers-500.data.arff.gz"))).toData()
  }

  describe("Papers 500 sample data") {
    it("should be read properly") {
      new Papers500 {
        data.size should equal (7718)
        sparseData.getNumOfDatacase should equal (7692)    // the value is obtained by run time
        wholeData.getTotalWeight should equal (7692)    //  same as the sparse data set
      }
    }

    it("should allow conversion from sparse data to data set") {
      new Papers500 {
        val converted = DataOps.convertToDataSet(sparseData)

        checkDataSets(converted, wholeData)
      }
    }

    it("should project to the first three variables properly") {
      new Papers500 {
        val vs = wholeData.getVariables
        val subset = new ArrayList(Arrays.asList(vs(0), vs(1), vs(2)))
        checkDataSets(DataOps.project(wholeData, subset), wholeData.project(subset))
      }
    }

    it("should project to the a subset of 10 variables properly") {
      new Papers500 {
        val vs = wholeData.getVariables
        val subset = new ArrayList(Arrays.asList(
          vs(100), vs(200), vs(300), vs(400), vs(499), vs(252), vs(325), vs(0), vs(12), vs(398)))
        checkDataSets(DataOps.project(wholeData, subset), wholeData.project(subset))
      }
    }

    it("should sample data appropriately") {
      new Papers500 {
        val size = 10
        val samples = DataOps.sampleWithReplacement(wholeData, size)
        samples.getTotalWeight should equal (size)
        samples.getNumberOfEntries should be >= 0
        samples.getNumberOfEntries should be <= 10
      }
    }

  }

  describe("Small data set") {
    it("should be read properly") {
      new SmallTestData {
        data.size should equal (10)
        sparseData.getNumOfDatacase should equal (7)    // the data cases with all zeros are discarded
        wholeData.getTotalWeight should equal (7)
        wholeData.getData.size should equal (3)
      }
    }

    it("should allow conversion from sparse data to data weights") {
      new SmallTestData {
        val converted = DataOps.convertToDataCases(sparseData).toArrayList()
        converted.size should equal (3)

        checkDataCase(converted.get(0), Array(0,1,0,0,0), 1)
        checkDataCase(converted.get(1), Array(1,0,0,0,1), 2)
        checkDataCase(converted.get(2), Array(1,1,1,0,1), 4)
      }
    }

    it("should allow conversion from sparse data to data set") {
      new SmallTestData {
        checkDataSets(DataOps.convertToDataSet(sparseData), wholeData)
      }
    }

    it("should project to the first  variable properly") {
      new SmallTestData {
        val vs = wholeData.getVariables
        val subset: ArrayList[org.latlab.util.Variable] =
          new ArrayList(Arrays.asList(vs(0)))
        checkDataSets(DataOps.project(wholeData, subset), wholeData.project(subset))
      }
    }

    it("should project to the first three variables properly") {
      new SmallTestData {
        val vs = wholeData.getVariables
        val subset: ArrayList[org.latlab.util.Variable] =
          new ArrayList(Arrays.asList(vs(0), vs(1), vs(2)))
        checkDataSets(DataOps.project(wholeData, subset), wholeData.project(subset))
      }
    }

    it("should project to the last four variables properly") {
      new SmallTestData {
        val vs = wholeData.getVariables
        val subset = new ArrayList(Arrays.asList(vs(2), vs(3), vs(1), vs(4)))
        checkDataSets(DataOps.project(wholeData, subset), wholeData.project(subset))
      }
    }

    it("should sample data appropriately") {
      new SmallTestData {
        val size = 10
        val samples = DataOps.sampleWithReplacement(wholeData, size)
        samples.getTotalWeight should equal (size)
        samples.getNumberOfEntries should be >= 0
        samples.getNumberOfEntries should be <= 10
      }
    }

  }

  def checkDataSets(subject: DataSet, target: DataSet) = {
    subject.getVariables should contain theSameElementsInOrderAs target.getVariables
    subject.getTotalWeight should equal (target.getTotalWeight)

    for (i <- 0 until target.getData.size()) {
      val c1 = subject.getData.get(i)
      val c2 = target.getData.get(i)

      c1.getStates should contain theSameElementsInOrderAs c2.getStates
      c1.getWeight should equal (c2.getWeight)
    }
  }

  def checkDataCase(target: DataCase, states: Array[Int], weight: Double) = {
    target.getStates should contain theSameElementsInOrderAs states
    target.getWeight should equal (weight)
  }
}
