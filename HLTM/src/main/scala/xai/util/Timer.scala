package xai.util

object Timer {
  def time[T](message: String)(block: => T): T = {
    val v0 = System.nanoTime()
    val res = block
    val v1 = System.nanoTime()
    println(s"Time for ${message}: ${(v1 - v0)/1e9d} seconds")
    res
  }
}
