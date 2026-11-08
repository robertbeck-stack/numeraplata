package app.numby.alpr

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max

// Icelandic license plates: 2 letters + 3 digits (AB123) or 3 letters + 2 digits (ABC12)
private val PLATE_RE = Regex("^([A-Z]{2}[0-9]{3}|[A-Z]{3}[0-9]{2})$")
private val STICKER_RE = Regex("^([A-Z0-9]{2})[0-9]{2}([A-Z0-9]{3})$")

private fun normalizeFive(s: String): String {
  if (s.length != 5) return s
  // Require at least one real letter in first 2 chars and one real digit in last 2 chars to avoid turning words into plates
  if (!s[0].isLetter() && !s[1].isLetter()) return s
  if (!s[3].isDigit() && !s[4].isDigit()) return s
  fun toLet(c: Char) = when (c) { '0' -> 'O'; '1' -> 'I'; '8' -> 'B'; else -> c }
  fun toDig(c: Char) = when (c) { 'O', 'Q', 'D' -> '0'; 'I', 'L' -> '1'; 'B' -> '8'; 'S' -> '5'; else -> c }
  return "${toLet(s[0])}${toLet(s[1])}${s[2]}${toDig(s[3])}${toDig(s[4])}"
}

fun extractPlate(raw: String): String? {
  val cleaned = raw.uppercase(Locale.ROOT).replace(Regex("[^A-Z0-9]"), "").removePrefix("IS")
  val norm5 = normalizeFive(cleaned)
  if (PLATE_RE.matches(norm5)) return norm5
  val m = STICKER_RE.matchEntire(cleaned)
  if (m != null) {
    val combined = normalizeFive(m.groupValues[1] + m.groupValues[2])
    if (PLATE_RE.matches(combined)) return combined
  }
  return null
}

// True if two 5-char plates are from the same car (<=1 char diff, or 4 chars match in order when a leading 'I' was missed)
fun isSameCar(a: String, b: String): Boolean {
  if (a.length != 5 || b.length != 5) return false
  if (a.zip(b).count { (x, y) -> x != y } <= 1) return true
  // Check if 4 of 5 characters appear in the same relative order (LCS >= 4, e.g. XLS53 vs IXL53)
  val dp = Array(6) { IntArray(6) }
  for (i in 1..5) {
    for (j in 1..5) {
      dp[i][j] = if (a[i - 1] == b[j - 1]) dp[i - 1][j - 1] + 1 else max(dp[i - 1][j], dp[i][j - 1])
    }
  }
  return dp[5][5] >= 4
}

data class ScanEntry(
  val time: String,
  val plate: String,
  val confidence: Float = 0f,
  val imgPath: String? = null,
  val timestampMs: Long = System.currentTimeMillis()
)

private fun loadSampledBitmap(path: String, reqW: Int, reqH: Int): Bitmap? {
  val file = File(path)
  if (!file.exists()) return null
  val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
  BitmapFactory.decodeFile(path, opts)
  var sample = 1
  if (opts.outHeight > reqH || opts.outWidth > reqW) {
    val halfH = opts.outHeight / 2
    val halfW = opts.outWidth / 2
    while ((halfH / sample) >= reqH && (halfW / sample) >= reqW) {
      sample *= 2
    }
  }
  return BitmapFactory.decodeFile(path, BitmapFactory.Options().apply { inSampleSize = sample })
}

private fun saveDebugShot(context: Context, bmp: Bitmap, box: Rect?, plate: String, ts: String): String {
  val copy = bmp.copy(Bitmap.Config.ARGB_8888, true)
  val canvas = android.graphics.Canvas(copy)
  val paint = Paint().apply {
    color = android.graphics.Color.GREEN
    style = Paint.Style.STROKE
    strokeWidth = 5f
    textSize = 36f
    isAntiAlias = true
  }
  if (box != null) {
    canvas.drawRect(box, paint)
    paint.style = Paint.Style.FILL
    canvas.drawText(plate, box.left.toFloat(), (box.top - 12).coerceAtLeast(48).toFloat(), paint)
  }
  val file = File(context.filesDir, "debug_${plate}_${System.currentTimeMillis()}.jpg")
  FileOutputStream(file).use { copy.compress(Bitmap.CompressFormat.JPEG, 95, it) }

  val values = ContentValues().apply {
    put(MediaStore.Images.Media.DISPLAY_NAME, "numby_${plate}_${ts.replace(":", "").replace(" ", "_")}.jpg")
    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Numby")
  }
  context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)?.let { uri ->
    context.contentResolver.openOutputStream(uri)?.use { copy.compress(Bitmap.CompressFormat.JPEG, 95, it) }
  }
  return file.absolutePath
}

private fun shareEntry(context: Context, entry: ScanEntry) {
  val text = "${entry.plate} (${entry.time})"
  val file = entry.imgPath?.let { File(it) }?.takeIf { it.exists() }
  val intent = Intent(Intent.ACTION_SEND).apply {
    putExtra(Intent.EXTRA_SUBJECT, entry.plate)
    putExtra(Intent.EXTRA_TEXT, text)
    if (file != null) {
      val uri = androidx.core.content.FileProvider.getUriForFile(
        context,
        "${context.packageName}.fileprovider",
        file
      )
      type = "image/jpeg"
      putExtra(Intent.EXTRA_STREAM, uri)
      addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    } else {
      type = "text/plain"
    }
  }
  context.startActivity(Intent.createChooser(intent, "Deila ${entry.plate}"))
}

fun formatDayHeader(dateStr: String): String {
  return try {
    val parsed = SimpleDateFormat("yyyy-MM-dd", Locale.ROOT).parse(dateStr) ?: return dateStr
    SimpleDateFormat("EEEE d. MMM yyyy", Locale("is", "IS")).format(parsed).replaceFirstChar { it.uppercase() }
  } catch (_: Exception) {
    dateStr
  }
}

val CyberDarkScheme = darkColorScheme(
  primary = Color(0xFF00E5FF),
  onPrimary = Color(0xFF050811),
  secondary = Color(0xFF00FF88),
  onSecondary = Color(0xFF050811),
  background = Color(0xFF060913),
  onBackground = Color(0xFFF1F5F9),
  surface = Color(0xFF0D1322),
  onSurface = Color(0xFFE2E8F0),
  surfaceVariant = Color(0xFF141D30),
  onSurfaceVariant = Color(0xFF94A3B8),
  outline = Color(0xFF1E293B)
)

@Composable
fun IcelandicPlateBadge(plate: String, modifier: Modifier = Modifier) {
  val formatted = if (plate.length == 5) {
    if (plate[2].isDigit()) "${plate.substring(0, 2)} ${plate.substring(2)}"
    else "${plate.substring(0, 3)} ${plate.substring(3)}"
  } else plate

  Surface(
    modifier = modifier.width(118.dp),
    shape = RoundedCornerShape(5.dp),
    color = Color(0xFFF8FAFC),
    border = BorderStroke(1.5.dp, Color(0xFF1E293B)),
    shadowElevation = 1.dp
  ) {
    Row(
      verticalAlignment = Alignment.CenterVertically,
      modifier = Modifier.padding(start = 3.5.dp, end = 5.dp, top = 2.dp, bottom = 2.dp)
    ) {
      Surface(
        color = Color(0xFF0051B4),
        shape = RoundedCornerShape(2.dp),
        modifier = Modifier.padding(end = 5.dp)
      ) {
        Column(
          horizontalAlignment = Alignment.CenterHorizontally,
          modifier = Modifier.padding(horizontal = 2.5.dp, vertical = 1.5.dp)
        ) {
          Text(
            text = "IS",
            color = Color.White,
            fontWeight = FontWeight.Black,
            fontSize = 8.sp,
            lineHeight = 9.sp
          )
        }
      }
      Text(
        text = formatted,
        color = Color(0xFF0F172A),
        fontWeight = FontWeight.ExtraBold,
        fontSize = 14.sp,
        letterSpacing = 1.sp,
        textAlign = TextAlign.Center,
        maxLines = 1,
        softWrap = false,
        modifier = Modifier.weight(1f)
      )
    }
  }
}

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    val logFile = File(filesDir, "plates.csv")
    setContent {
      MaterialTheme(colorScheme = CyberDarkScheme) {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
          PlateScannerScreen(logFile)
        }
      }
    }
  }
}

@OptIn(ExperimentalGetImage::class)
@Composable
fun PlateScannerScreen(logFile: File) {
  val context = LocalContext.current
  val lifecycleOwner = LocalLifecycleOwner.current
  var hasCamPerm by remember {
    mutableStateOf(
      ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    )
  }
  val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasCamPerm = it }
  LaunchedEffect(Unit) { if (!hasCamPerm) launcher.launch(Manifest.permission.CAMERA) }

  val logs = remember {
    mutableStateListOf<ScanEntry>().apply {
      if (logFile.exists()) {
        logFile.readLines().reversed().forEach { line ->
          val parts = line.split(",")
          if (parts.size >= 4) {
            val conf = parts[2].removeSuffix("%").toFloatOrNull()?.let { if (it > 1f) it / 100f else it } ?: 1f
            add(ScanEntry(parts[0], parts[1], conf, parts[3].takeIf { it.isNotEmpty() }, 0L))
          } else if (parts.size >= 2) {
            add(ScanEntry(parts[0], parts[1], 1f, parts.getOrNull(2)?.takeIf { it.isNotEmpty() }, 0L))
          }
        }
      }
    }
  }
  var showBoxes by remember { mutableStateOf(true) }
  var debugShots by remember { mutableStateOf(true) }
  var newestFirst by remember { mutableStateOf(true) }
  var selectedEntry by remember { mutableStateOf<ScanEntry?>(null) }
  var detectedBoxes by remember { mutableStateOf<List<Rect>>(emptyList()) }
  var frameSize by remember { mutableStateOf(1 to 1) }

  fun rewriteCsv() {
    logFile.writeText(logs.reversed().joinToString("\n") { "${it.time},${it.plate},${(it.confidence * 100).toInt()}%,${it.imgPath ?: ""}" } + if (logs.isNotEmpty()) "\n" else "")
  }

  Column(modifier = Modifier.fillMaxSize().safeDrawingPadding()) {
    if (hasCamPerm) {
      Box(modifier = Modifier.fillMaxWidth().height(245.dp).clipToBounds()) {
        AndroidView(
          modifier = Modifier.fillMaxSize(),
          factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
              scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            val yoloDetector = YoloPlateDetector(ctx)
            val executor = Executors.newSingleThreadExecutor()
            val saveExecutor = Executors.newSingleThreadExecutor()
            val resSelector = ResolutionSelector.Builder()
              .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
              .setResolutionStrategy(
                ResolutionStrategy(android.util.Size(1920, 1080), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
              )
              .build()

            ProcessCameraProvider.getInstance(ctx).addListener({
              val provider = ProcessCameraProvider.getInstance(ctx).get()
              val preview = Preview.Builder()
                .setResolutionSelector(resSelector)
                .build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
              val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(resSelector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { ia ->
                  ia.setAnalyzer(executor) { proxy ->
                    val vw = previewView.width
                    val vh = previewView.height
                    if (vw <= 0 || vh <= 0) {
                      proxy.close()
                      return@setAnalyzer
                    }
                    val rawBmp = proxy.toBitmap()
                    val rot = proxy.imageInfo.rotationDegrees
                    val rotated = if (rot != 0) {
                      val m = Matrix().apply { postRotate(rot.toFloat()) }
                      Bitmap.createBitmap(rawBmp, 0, 0, rawBmp.width, rawBmp.height, m, true)
                    } else rawBmp

                    val scale = max(vw.toFloat() / rotated.width, vh.toFloat() / rotated.height)
                    val cropW = (vw / scale).toInt().coerceIn(1, rotated.width)
                    val cropH = (vh / scale).toInt().coerceIn(1, rotated.height)
                    val cropX = ((rotated.width - cropW) / 2).coerceAtLeast(0)
                    val cropY = ((rotated.height - cropH) / 2).coerceAtLeast(0)
                    val visibleBmp = Bitmap.createBitmap(rotated, cropX, cropY, cropW, cropH)

                    val yoloDetections = yoloDetector.detect(visibleBmp, confThreshold = 0.35f)

                    if (yoloDetections.isNotEmpty()) {
                      val boxes = mutableListOf<Rect>()
                      val now = System.currentTimeMillis()
                      var pendingTasks = yoloDetections.size

                      for (det in yoloDetections) {
                        val dBox = det.box
                        val padX = (dBox.width() * 0.10f).toInt().coerceAtLeast(4)
                        val padY = (dBox.height() * 0.20f).toInt().coerceAtLeast(4)
                        val cLeft = (dBox.left - padX).coerceAtLeast(0)
                        val cTop = (dBox.top - padY).coerceAtLeast(0)
                        val cRight = (dBox.right + padX).coerceAtMost(cropW)
                        val cBottom = (dBox.bottom + padY).coerceAtMost(cropH)
                        val cW = cRight - cLeft
                        val cH = cBottom - cTop

                        if (cW > 10 && cH > 10) {
                          val plateCrop = Bitmap.createBitmap(visibleBmp, cLeft, cTop, cW, cH)
                          val cropInput = InputImage.fromBitmap(plateCrop, 0)

                          recognizer.process(cropInput)
                            .addOnSuccessListener { visionText ->
                              for (block in visionText.textBlocks) {
                                for (line in block.lines) {
                                  val maxH = line.elements.maxOfOrNull { it.boundingBox?.height() ?: 0 } ?: 0
                                  val tallElems = line.elements.filter { (it.boundingBox?.height() ?: maxH) >= maxH * 0.6 }
                                  val tallText = tallElems.joinToString("") { it.text }
                                  val plate = extractPlate(tallText) ?: extractPlate(line.text)

                                  if (plate != null) {
                                    boxes.add(dBox)
                                    val conf = det.confidence
                                    val fullBox = Rect(dBox.left + cropX, dBox.top + cropY, dBox.right + cropX, dBox.bottom + cropY)

                                    val matchIdx = logs.indexOfFirst {
                                      now - it.timestampMs < 25_000L && (it.plate == plate || isSameCar(it.plate, plate))
                                    }
                                    if (matchIdx == -1) {
                                      val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(now))
                                      val entry = ScanEntry(ts, plate, conf, null, now)
                                      logs.add(0, entry)
                                      rewriteCsv()
                                      if (debugShots) {
                                        val capturedBmp = rotated.copy(Bitmap.Config.ARGB_8888, false)
                                        saveExecutor.execute {
                                          val shotPath = saveDebugShot(ctx, capturedBmp, fullBox, plate, ts)
                                          val idx = logs.indexOfFirst { it.time == ts && it.plate == plate }
                                          if (idx != -1) {
                                            logs[idx] = logs[idx].copy(imgPath = shotPath)
                                            rewriteCsv()
                                          }
                                        }
                                      }
                                    } else if (conf > logs[matchIdx].confidence && logs[matchIdx].plate != plate) {
                                      val old = logs[matchIdx]
                                      logs[matchIdx] = old.copy(plate = plate, confidence = conf, timestampMs = now)
                                      rewriteCsv()
                                      if (debugShots) {
                                        val capturedBmp = rotated.copy(Bitmap.Config.ARGB_8888, false)
                                        saveExecutor.execute {
                                          val shotPath = saveDebugShot(ctx, capturedBmp, fullBox, plate, old.time)
                                          val idx = logs.indexOfFirst { it.time == old.time && it.plate == plate }
                                          if (idx != -1) {
                                            logs[idx] = logs[idx].copy(imgPath = shotPath)
                                            rewriteCsv()
                                          }
                                        }
                                      }
                                    }
                                  }
                                }
                              }
                            }
                            .addOnCompleteListener {
                              pendingTasks--
                              if (pendingTasks <= 0) {
                                frameSize = cropW to cropH
                                detectedBoxes = boxes
                                proxy.close()
                              }
                            }
                        } else {
                          pendingTasks--
                          if (pendingTasks <= 0) {
                            frameSize = cropW to cropH
                            detectedBoxes = boxes
                            proxy.close()
                          }
                        }
                      }
                    } else {
                      // Fallback: full frame ML Kit OCR if YOLO didn't detect bounding box
                      val img = InputImage.fromBitmap(visibleBmp, 0)
                      recognizer.process(img)
                        .addOnSuccessListener { visionText ->
                          val boxes = mutableListOf<Rect>()
                          val now = System.currentTimeMillis()
                          for (block in visionText.textBlocks) {
                            for (line in block.lines) {
                              val maxH = line.elements.maxOfOrNull { it.boundingBox?.height() ?: 0 } ?: 0
                              val tallElems = line.elements.filter { (it.boundingBox?.height() ?: maxH) >= maxH * 0.6 }
                              val tallText = tallElems.joinToString("") { it.text }
                              val tallPlate = extractPlate(tallText)
                              val plate = tallPlate ?: extractPlate(line.text)
                              val usedElems = if (tallPlate != null) tallElems else line.elements
                              val conf = usedElems.mapNotNull { it.confidence }.average().toFloat().takeIf { !it.isNaN() }
                                ?: line.confidence ?: 0.5f
                              if (plate != null && conf >= 0.45f) {
                                val rawBox = (if (tallPlate != null) {
                                  tallElems.mapNotNull { it.boundingBox }.reduceOrNull { acc, r -> Rect(acc).apply { union(r) } }
                                } else null) ?: line.boundingBox
                                val box = rawBox?.let {
                                  val padX = (it.width() * 0.08f).toInt().coerceAtLeast(6)
                                  val padY = (it.height() * 0.18f).toInt().coerceAtLeast(6)
                                  Rect(
                                    (it.left - padX).coerceAtLeast(0),
                                    (it.top - padY).coerceAtLeast(0),
                                    (it.right + padX).coerceAtMost(cropW),
                                    (it.bottom + padY).coerceAtMost(cropH)
                                  )
                                }
                                box?.let { boxes.add(it) }

                                val fullBox = box?.let { Rect(it.left + cropX, it.top + cropY, it.right + cropX, it.bottom + cropY) }
                                val matchIdx = logs.indexOfFirst {
                                  now - it.timestampMs < 25_000L && (it.plate == plate || isSameCar(it.plate, plate))
                                }
                                if (matchIdx == -1) {
                                  val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(now))
                                  val entry = ScanEntry(ts, plate, conf, null, now)
                                  logs.add(0, entry)
                                  rewriteCsv()
                                  if (debugShots) {
                                    val capturedBmp = rotated.copy(Bitmap.Config.ARGB_8888, false)
                                    saveExecutor.execute {
                                      val shotPath = saveDebugShot(ctx, capturedBmp, fullBox, plate, ts)
                                      val idx = logs.indexOfFirst { it.time == ts && it.plate == plate }
                                      if (idx != -1) {
                                        logs[idx] = logs[idx].copy(imgPath = shotPath)
                                        rewriteCsv()
                                      }
                                    }
                                  }
                                } else if (conf > logs[matchIdx].confidence && logs[matchIdx].plate != plate) {
                                  val old = logs[matchIdx]
                                  logs[matchIdx] = old.copy(plate = plate, confidence = conf, timestampMs = now)
                                  rewriteCsv()
                                  if (debugShots) {
                                    val capturedBmp = rotated.copy(Bitmap.Config.ARGB_8888, false)
                                    saveExecutor.execute {
                                      val shotPath = saveDebugShot(ctx, capturedBmp, fullBox, plate, old.time)
                                      val idx = logs.indexOfFirst { it.time == old.time && it.plate == plate }
                                      if (idx != -1) {
                                        logs[idx] = logs[idx].copy(imgPath = shotPath)
                                        rewriteCsv()
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                          frameSize = cropW to cropH
                          detectedBoxes = boxes
                        }
                        .addOnCompleteListener { proxy.close() }
                    }
                  }
                }
              provider.unbindAll()
              val cam = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
              cam.cameraControl.setZoomRatio(1.5f)
            }, ContextCompat.getMainExecutor(ctx))
            previewView
          }
        )
        // Cyber HUD gradient vignette at bottom of camera feed
        Box(
          modifier = Modifier
            .fillMaxWidth()
            .height(48.dp)
            .align(Alignment.BottomCenter)
            .background(
              Brush.verticalGradient(
                listOf(Color.Transparent, Color(0xFF060913))
              )
            )
        )

        // Top Cyber HUD Diagnostic Strip
        Row(
          modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 12.dp, vertical = 10.dp)
            .align(Alignment.TopCenter),
          horizontalArrangement = Arrangement.SpaceBetween,
          verticalAlignment = Alignment.CenterVertically
        ) {
          Surface(
            color = Color(0xDD060913),
            shape = RoundedCornerShape(20.dp),
            border = BorderStroke(1.dp, Color(0xFF00FF88).copy(alpha = 0.5f))
          ) {
            Row(
              verticalAlignment = Alignment.CenterVertically,
              horizontalArrangement = Arrangement.spacedBy(6.dp),
              modifier = Modifier.padding(horizontal = 10.dp, vertical = 4.dp)
            ) {
              Box(
                modifier = Modifier
                  .size(8.dp)
                  .background(Color(0xFF00FF88), RoundedCornerShape(4.dp))
              )
              Text(
                "YOLOv9 VIRKT",
                color = Color(0xFF00FF88),
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
              )
            }
          }

          Surface(
            color = Color(0xDD060913),
            shape = RoundedCornerShape(8.dp),
            border = BorderStroke(1.dp, Color(0xFF00E5FF).copy(alpha = 0.35f))
          ) {
            Text(
              "LIVE • 1080p",
              color = Color(0xFF00E5FF),
              fontSize = 11.sp,
              fontWeight = FontWeight.SemiBold,
              modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
            )
          }
        }

        if (showBoxes && detectedBoxes.isNotEmpty()) {
          Canvas(modifier = Modifier.fillMaxSize()) {
            val (fw, fh) = frameSize
            val sx = size.width / fw
            val sy = size.height / fh
            for (rect in detectedBoxes) {
              val left = rect.left * sx
              val top = rect.top * sy
              val right = rect.right * sx
              val bottom = rect.bottom * sy
              val w = right - left
              val h = bottom - top
              val cornerLen = minOf(w, h) * 0.28f

              // Semi-transparent glowing target fill
              drawRect(
                color = Color(0x2200E5FF),
                topLeft = Offset(left, top),
                size = Size(w, h)
              )

              // Subtle bounding border
              drawRect(
                color = Color(0x4400E5FF),
                topLeft = Offset(left, top),
                size = Size(w, h),
                style = Stroke(width = 1.5f)
              )

              // Neon Cyber Reticle Corner L-brackets
              val bracketColor = Color(0xFF00FF88)
              val strokeW = 6f

              // Top-Left
              drawLine(bracketColor, Offset(left, top), Offset(left + cornerLen, top), strokeWidth = strokeW)
              drawLine(bracketColor, Offset(left, top), Offset(left, top + cornerLen), strokeWidth = strokeW)

              // Top-Right
              drawLine(bracketColor, Offset(right, top), Offset(right - cornerLen, top), strokeWidth = strokeW)
              drawLine(bracketColor, Offset(right, top), Offset(right, top + cornerLen), strokeWidth = strokeW)

              // Bottom-Left
              drawLine(bracketColor, Offset(left, bottom), Offset(left + cornerLen, bottom), strokeWidth = strokeW)
              drawLine(bracketColor, Offset(left, bottom), Offset(left, bottom - cornerLen), strokeWidth = strokeW)

              // Bottom-Right
              drawLine(bracketColor, Offset(right, bottom), Offset(right - cornerLen, bottom), strokeWidth = strokeW)
              drawLine(bracketColor, Offset(right, bottom), Offset(right, bottom - cornerLen), strokeWidth = strokeW)
            }
          }
        }
      }
    } else {
      Text("Vantar leyfi fyrir myndavél", modifier = Modifier.padding(16.dp))
    }

    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 14.dp, vertical = 4.dp)) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
      ) {
        Text(
          "Skráð atvik",
          fontWeight = FontWeight.Bold,
          fontSize = 17.sp,
          color = Color.White
        )
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
          OutlinedButton(
            enabled = logs.isNotEmpty(),
            onClick = {
              logs.clear()
              logFile.delete()
            },
            colors = ButtonDefaults.outlinedButtonColors(
              contentColor = Color(0xFFF43F5E)
            ),
            border = BorderStroke(1.dp, Color(0xFFF43F5E).copy(alpha = 0.5f)),
            shape = RoundedCornerShape(8.dp),
            contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
            modifier = Modifier.height(32.dp)
          ) {
            Text("Hreinsa", maxLines = 1, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
          }
          Button(
            enabled = logs.isNotEmpty(),
            onClick = {
              val ordered = if (newestFirst) logs else logs.reversed()
              val csv = "Dagur,Tími,Bílnúmer,Öryggi\n" + ordered.joinToString("\n") { entry ->
                val parts = entry.time.split(" ")
                val confPct = "${(entry.confidence * 100).toInt()}%"
                "${parts.getOrElse(0) { "" }},${parts.getOrElse(1) { entry.time }},${entry.plate},$confPct"
              }
              val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_SUBJECT, "Atvik (${logs.size})")
                putExtra(Intent.EXTRA_TEXT, csv)
              }
              context.startActivity(Intent.createChooser(intent, "Deila atvikum"))
            },
            colors = ButtonDefaults.buttonColors(
              containerColor = Color(0xFF00E5FF),
              contentColor = Color(0xFF050811)
            ),
            shape = RoundedCornerShape(8.dp),
            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 4.dp),
            modifier = Modifier.height(32.dp)
          ) {
            Text("Deila CSV", fontWeight = FontWeight.Bold, maxLines = 1, fontSize = 11.sp)
          }
        }
      }
      Spacer(modifier = Modifier.height(4.dp))
      Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        FilterChip(
          selected = showBoxes,
          onClick = { showBoxes = !showBoxes },
          label = { Text("Rammi", fontSize = 11.sp) },
          colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = Color(0xFF00E5FF).copy(alpha = 0.2f),
            selectedLabelColor = Color(0xFF00E5FF)
          ),
          border = FilterChipDefaults.filterChipBorder(
            enabled = true,
            selected = showBoxes,
            borderColor = if (showBoxes) Color(0xFF00E5FF) else Color(0xFF1E293B),
            selectedBorderColor = Color(0xFF00E5FF)
          ),
          modifier = Modifier.height(28.dp)
        )
        FilterChip(
          selected = debugShots,
          onClick = { debugShots = !debugShots },
          label = { Text("Skjáskot", fontSize = 11.sp) },
          colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = Color(0xFF00E5FF).copy(alpha = 0.2f),
            selectedLabelColor = Color(0xFF00E5FF)
          ),
          border = FilterChipDefaults.filterChipBorder(
            enabled = true,
            selected = debugShots,
            borderColor = if (debugShots) Color(0xFF00E5FF) else Color(0xFF1E293B),
            selectedBorderColor = Color(0xFF00E5FF)
          ),
          modifier = Modifier.height(28.dp)
        )
        FilterChip(
          selected = !newestFirst,
          onClick = { newestFirst = !newestFirst },
          label = { Text(if (newestFirst) "Nýjast efst" else "Elst efst", fontSize = 11.sp) },
          colors = FilterChipDefaults.filterChipColors(
            selectedContainerColor = Color(0xFF00E5FF).copy(alpha = 0.2f),
            selectedLabelColor = Color(0xFF00E5FF)
          ),
          border = FilterChipDefaults.filterChipBorder(
            enabled = true,
            selected = !newestFirst,
            borderColor = if (!newestFirst) Color(0xFF00E5FF) else Color(0xFF1E293B),
            selectedBorderColor = Color(0xFF00E5FF)
          ),
          modifier = Modifier.height(28.dp)
        )
      }
    }
    val orderedLogs = if (newestFirst) logs else logs.reversed()
    val groupedByDay = orderedLogs.groupBy { it.time.substringBefore(" ") }
    LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
      groupedByDay.forEach { (dayStr, dayEntries) ->
        item(key = "day_$dayStr") {
          Row(
            modifier = Modifier
              .fillMaxWidth()
              .padding(start = 4.dp, end = 4.dp, top = 8.dp, bottom = 4.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
          ) {
            Text(
              formatDayHeader(dayStr).uppercase(),
              fontWeight = FontWeight.Bold,
              fontSize = 11.sp,
              letterSpacing = 1.sp,
              color = Color(0xFF00E5FF)
            )
            Text(
              "${dayEntries.size} bílar",
              fontSize = 11.sp,
              color = Color(0xFF64748B)
            )
          }
        }
        items(dayEntries) { entry ->
          val timeNoSec = entry.time.substringAfter(" ").take(5)
          Card(
            modifier = Modifier
              .fillMaxWidth()
              .padding(vertical = 2.dp)
              .clickable { selectedEntry = entry },
            shape = RoundedCornerShape(10.dp),
            colors = CardDefaults.cardColors(
              containerColor = Color(0xFF0D1322)
            ),
            border = BorderStroke(1.dp, Color(0xFF1A2333))
          ) {
            Row(
              modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 7.dp),
              horizontalArrangement = Arrangement.SpaceBetween,
              verticalAlignment = Alignment.CenterVertically
            ) {
              Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(12.dp)
              ) {
                IcelandicPlateBadge(entry.plate)
                val pct = (entry.confidence * 100).toInt()
                val badgeColor = if (pct >= 85) Color(0xFF00FF88) else Color(0xFFFFB300)
                Surface(
                  color = badgeColor.copy(alpha = 0.12f),
                  shape = RoundedCornerShape(4.dp),
                  border = BorderStroke(0.5.dp, badgeColor.copy(alpha = 0.6f))
                ) {
                  Text(
                    text = "$pct%",
                    color = badgeColor,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1,
                    softWrap = false,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
                  )
                }
              }
              Text(
                text = timeNoSec,
                fontSize = 13.sp,
                fontWeight = FontWeight.SemiBold,
                color = Color(0xFF94A3B8),
                maxLines = 1,
                softWrap = false
              )
            }
          }
        }
      }
    }

    selectedEntry?.let { entry ->
      AlertDialog(
        onDismissRequest = { selectedEntry = null },
        containerColor = Color(0xFF0D1322),
        titleContentColor = Color.White,
        textContentColor = Color(0xFFE2E8F0),
        title = {
          Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
          ) {
            IcelandicPlateBadge(entry.plate)
            val pct = (entry.confidence * 100).toInt()
            val badgeColor = if (pct >= 85) Color(0xFF00FF88) else Color(0xFFFFB300)
            Surface(
              color = badgeColor.copy(alpha = 0.15f),
              shape = RoundedCornerShape(4.dp),
              border = BorderStroke(1.dp, badgeColor)
            ) {
              Text(
                "$pct% Öryggi",
                color = badgeColor,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp)
              )
            }
          }
        },
        text = {
          Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(entry.time, color = Color(0xFF94A3B8), fontSize = 13.sp)
            val bmp = remember(entry.imgPath) {
              entry.imgPath?.takeIf { File(it).exists() }?.let { loadSampledBitmap(it, 1080, 1080) }
            }
            if (bmp != null) {
              Surface(
                shape = RoundedCornerShape(8.dp),
                border = BorderStroke(1.dp, Color(0xFF1E293B))
              ) {
                Image(
                  bitmap = bmp.asImageBitmap(),
                  contentDescription = entry.plate,
                  modifier = Modifier.fillMaxWidth()
                )
              }
            } else {
              Text("Engin mynd vistuð fyrir þessa færslu.", color = Color(0xFF64748B))
            }
          }
        },
        confirmButton = {
          Button(
            onClick = { shareEntry(context, entry) },
            colors = ButtonDefaults.buttonColors(
              containerColor = Color(0xFF00E5FF),
              contentColor = Color(0xFF050811)
            ),
            shape = RoundedCornerShape(6.dp)
          ) {
            Text("Deila", fontWeight = FontWeight.Bold)
          }
        },
        dismissButton = {
          TextButton(onClick = { selectedEntry = null }) {
            Text("Loka", color = Color(0xFF94A3B8))
          }
        }
      )
    }
  }
}
