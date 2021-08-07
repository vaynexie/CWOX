package org.latlab.learner


import java.util
import java.util.Collections

import cern.jet.random.Uniform
import org.latlab.util.DataSet.DataCase
import org.latlab.util.{DataSet, Variable}

import scala.collection.JavaConverters._
import scala.collection.Searching._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object DataOps {
  type States = Array[Int]

  def addToList(cases: util.ArrayList[DataCase], newCase: DataCase) = {
    //    println(s"add single case to list with ${cases.size()}")

    val index = Collections.binarySearch(cases, newCase)
    val inserted = if (index < 0) {
      cases.add(-index - 1, newCase)
      true
    } else {
      val c = cases.get(index)
      c.setWeight(c.getWeight + newCase.getWeight)
      false
    }

    inserted
  }

  def mergeToFirst(c1: DataCase, c2: DataCase): DataCase = {
    c1.setWeight(c1.getWeight + c2.getWeight)
    c1
  }

  /**
   * It merges two lists.  It assumes that the list of data cases in
   * the two lists are sorted with no duplicates.
   */
  def mergeLists(x: util.ArrayList[DataCase], y: util.ArrayList[DataCase]): util.ArrayList[DataCase] = {
    //    println(s"merge with ${x.size()} and ${y.size()}")

    val result = new util.ArrayList[DataCase](x.size() + y.size())

    var i = 0
    var j = 0
    while (i < x.size() && j < y.size()) {
      val c1 = x.get(i)
      val c2 = y.get(j)

      val cmp = c1.compareTo(c2)
      if (cmp < 0) {
        result.add(c1)
        i += 1
      } else if (cmp > 0) {
        result.add(c2)
        j += 1
      } else {
        result.add(mergeToFirst(c1, c2))
        i += 1
        j += 1
      }
    }

    def addRemaining(source: util.ArrayList[DataCase], start: Int) = {
      for (k <- start until source.size()) {
        result.add(source.get(k))
      }
    }

    addRemaining(x, i)
    addRemaining(y, j)

    result
  }

  sealed trait DataCases {
    def missing(): Boolean

    def totalWeights(): Double

    def get(i: Int): DataCase

    def size(): Int

    def toArrayList(): util.ArrayList[DataCase]

    def buildDataSet(variables: Array[Variable]): DataSet = {
      val result = new DataSet(variables)
      result.setDataCases(toArrayList(), missing(), totalWeights())
      result
    }

    def buildDataSet(dataSet: DataSet): DataSet = {
      dataSet.setDataCases(toArrayList(), missing(), totalWeights())
      dataSet
    }

    def add(other: DataCases): Multiple
  }

  object DataCases {
    def merge(x: DataCases, y: DataCases): Multiple = {
      // the checking makes sure that an Singleton object will not
      // need to add another Multiple object
      if (x.size() == 1)
        y.add(x)
      else
        x.add(y)
    }
  }

  case class Singleton(statesOp: () => (States, Double)) extends DataCases {
    private lazy val (states, _totalWeight) = statesOp()

    override def missing: Boolean = checkMissing(states)

    override def totalWeights: Double = _totalWeight

    override def get(i: Int) =
      if (i == 0) DataCase.construct(states, _totalWeight)
      else throw new IllegalArgumentException

    override def toArrayList(): util.ArrayList[DataCase] = {
      val result = new util.ArrayList[DataCase](1)
      result.add(get(0))
      result
    }

    override def size() = 1

    override def add(other: DataCases): Multiple = {
      val c1 = get(0)
      val c2 = other.get(0)

      val cases = new util.ArrayList[DataCase]()
      val cmp = c1.compareTo(c2)

      if (cmp < 0) {
        cases.add(c1)
        cases.add(c2)
      } else if (cmp > 0) {
        cases.add(c2)
        cases.add(c1)
      } else
        cases.add(mergeToFirst(c1, c2))

      Multiple(cases, missing || other.missing(), totalWeights + other.totalWeights())
    }

  }

  case class Multiple(private var cases: util.ArrayList[DataCase] = new util.ArrayList[DataCase](),
                      private var _missing: Boolean = false,
                      private var _totalWeights: Double = 0.0) extends DataCases {

    override def missing(): Boolean = _missing

    override def totalWeights(): Double = _totalWeights

    override def toArrayList(): util.ArrayList[DataCase] = cases

    override def get(i: Int) = cases.get(i)

    override def size() = cases.size()

    def add(newCase: DataCase): Multiple = {

      val inserted = addToList(cases, newCase)
      if (inserted) _missing ||= checkMissing(newCase.getStates)
      _totalWeights += newCase.getWeight
      this
    }

    def add(states: States, additionalWeight: Double): Multiple = {
      add(DataCase.construct(states, additionalWeight))
    }

    override def add(other: DataCases): Multiple = {
      if (other.size() == 1) {
        add(other.get(0))
      } else {
        cases = mergeLists(cases, other.toArrayList())
        _missing ||= other.missing()
        _totalWeights += other.totalWeights()
        this
      }
    }
  }

  def convertToDataCases(sparseData: SparseDataSet): DataCases = {
    val size = sparseData.getNumOfDatacase
    val order = Range(0, size)
    convertToDataCases(sparseData, size, 0, order, getInternalToExternalIDMapping(sparseData))
  }

  def convertToDataCases(sparseData: SparseDataSet,
                         batchSize: Int, start: Int,
                         order: IndexedSeq[Int],
                         intToExtID: Map[Integer, Int]): DataCases = {
    val length = sparseData._VariablesSet.size

    def getStates(i: Int) = {
      val states = Array.fill(length)(0)

      val row = sparseData.SDataSet.userMatrix.get(order(i))

      // Filling in the positive entries
      val iter = row.iterator
      while (iter.hasNext) {
        val internal_ID = iter.nextInt // the id of the item
        states(intToExtID(internal_ID)) = 1
      }

      (states, 1.0)
    }

    convertToDataCases(batchSize, start, getStates)
  }

  def convertToDataCases(batchSize: Int, start: Int,
                         retrieve: (Int) => (States, Double)): DataCases = {
    (start until start + batchSize)
      .map(i => Singleton(() => retrieve(i)).asInstanceOf[DataCases])
      .par
      .reduce(DataCases.merge)
  }


  def checkMissing(states: States) = states.exists(_ == DataSet.MISSING_VALUE)

  /**
   * Maps the internal ID to external ID based on the implementation in SparseDataSet
   */
  def getInternalToExternalIDMapping(sparseDataSet: SparseDataSet): Map[Integer, Int] = {
    def internalToExternal(internal: Integer): Int =
      sparseDataSet._mapNameToIndex.get(sparseDataSet._item_mapping.toOriginalID(internal))

    sparseDataSet.SDataSet.allItems.asScala.map(id => id -> internalToExternal(id)).toMap
  }

  def convertToDataSet(sparseData: SparseDataSet): DataSet = {
    convertToDataCases(sparseData).buildDataSet(sparseData._VariablesSet)
  }

  def project(dataSet: DataSet, subset: util.List[Variable]): DataSet = {
    // enforce ordering of the variables
    val result = new DataSet(subset.toArray(Array.ofDim[Variable](subset.size())))

    val map = result.getVariables.map(
      v => dataSet.getVariables.indexWhere(_.getName == v.getName))

    def getProjection(i: Int): (States, Double) = {
      val states = Array.fill(subset.size())(0)
      val c = dataSet.getData.get(i)
      (map.map(c.getStates()).toArray, c.getWeight)
    }

    val cases = convertToDataCases(dataSet.getData.size, 0, getProjection)
    cases.buildDataSet(result)
  }

  def sampleWithReplacement(dataSet: DataSet, sampleSize: Int): DataSet = {
    val random = new Random()
    sampleWithReplacement(dataSet, sampleSize, random.nextDouble)
  }

  /**
   * Generate the training data of the given size from the current data. The
   * method is sample from the current data with replacement. No side effect
   * on the input.
   */
  def sampleWithReplacement(dataSet: DataSet, sampleSize: Int, generator: () => Double): DataSet = {
    val cases = dataSet.getData

    val cummulativeWeights =
      cases.asScala.map(_.getWeight).scanLeft(0.0)(_ + _).toIndexedSeq

    def draw(): States = {
      val threshold = generator() * dataSet.getTotalWeight
      cummulativeWeights.search(threshold) match {
        case Found(i) =>
          cases.get(i).getStates
        case InsertionPoint(i) =>
          cases.get(i - 1).getStates
      }
    }

    convertToDataCases(sampleSize, 0, _ => (draw(), 1.0)).buildDataSet(dataSet.getVariables)
  }
}
