package app.numby.alpr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

data class PlateDetection(val box: Rect, val confidence: Float)

class YoloPlateDetector(context: Context) {
  private val env = OrtEnvironment.getEnvironment()
  private val session: OrtSession

  init {
    val modelBytes = context.assets.open("yolo_plate.onnx").use { it.readBytes() }
    session = env.createSession(modelBytes, OrtSession.SessionOptions())
  }

  fun detect(bitmap: Bitmap, confThreshold: Float = 0.40f): List<PlateDetection> {
    val srcW = bitmap.width
    val srcH = bitmap.height
    if (srcW <= 0 || srcH <= 0) return emptyList()

    val targetSize = 384
    val r = min(targetSize.toFloat() / srcW, targetSize.toFloat() / srcH)
    val newW = (srcW * r).toInt()
    val newH = (srcH * r).toInt()
    val padX = (targetSize - newW) / 2
    val padY = (targetSize - newH) / 2

    // Letterbox onto 384x384 canvas
    val letterboxBmp = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888)
    val canvas = Canvas(letterboxBmp)
    canvas.drawColor(Color.rgb(114, 114, 114)) // standard YOLO grey padding
    val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
    canvas.drawBitmap(scaled, padX.toFloat(), padY.toFloat(), Paint(Paint.FILTER_BITMAP_FLAG))

    // Convert to CHW float buffer normalized to 0.0 - 1.0 (RGB)
    val pixels = IntArray(targetSize * targetSize)
    letterboxBmp.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)

    val floatBuffer = FloatBuffer.allocate(3 * targetSize * targetSize)
    val area = targetSize * targetSize
    val gOffset = area
    val bOffset = 2 * area

    val channelR = FloatArray(area)
    val channelG = FloatArray(area)
    val channelB = FloatArray(area)

    for (i in 0 until area) {
      val px = pixels[i]
      channelR[i] = ((px shr 16) and 0xFF) / 255.0f
      channelG[i] = ((px shr 8) and 0xFF) / 255.0f
      channelB[i] = (px and 0xFF) / 255.0f
    }
    floatBuffer.put(channelR)
    floatBuffer.put(channelG)
    floatBuffer.put(channelB)
    floatBuffer.flip()

    val tensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong()))
    val results = mutableListOf<PlateDetection>()

    tensor.use { inputTensor ->
      session.run(mapOf("images" to inputTensor)).use { output ->
        @Suppress("UNCHECKED_CAST")
        val outputTensor = output.get(0) as OnnxTensor
        val raw = outputTensor.value as Array<FloatArray> // shape [N, 7]

        for (row in raw) {
          if (row.size < 7) continue
          val score = row[6]
          if (score < confThreshold) continue

          val x1 = row[1]
          val y1 = row[2]
          val x2 = row[3]
          val y2 = row[4]

          val origX1 = ((x1 - padX) / r).toInt().coerceIn(0, srcW)
          val origY1 = ((y1 - padY) / r).toInt().coerceIn(0, srcH)
          val origX2 = ((x2 - padX) / r).toInt().coerceIn(0, srcW)
          val origY2 = ((y2 - padY) / r).toInt().coerceIn(0, srcH)

          if (origX2 > origX1 && origY2 > origY1) {
            results.add(PlateDetection(Rect(origX1, origY1, origX2, origY2), score))
          }
        }
      }
    }
    return results
  }

  fun close() {
    session.close()
    env.close()
  }
}
