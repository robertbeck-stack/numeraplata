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
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
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
      val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, file.name)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/Numby")
      }
      val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
      if (uri != null) {
        context.contentResolver.openOutputStream(uri)?.use { out -> file.inputStream().use { it.copyTo(out) } }
        type = "image/jpeg"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
      } else {
        type = "text/plain"
      }
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

class MainActivity : ComponentActivity() {
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    enableEdgeToEdge()
    val logFile = File(filesDir, "plates.csv")
    setContent {
      MaterialTheme(colorScheme = darkColorScheme()) {
        Surface(modifier = Modifier.fillMaxSize()) {
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
          if (parts.size >= 2) add(ScanEntry(parts[0], parts[1], 1f, parts.getOrNull(2)?.takeIf { it.isNotEmpty() }, 0L))
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
    logFile.writeText(logs.reversed().joinToString("\n") { "${it.time},${it.plate},${it.imgPath ?: ""}" } + if (logs.isNotEmpty()) "\n" else "")
  }

  Column(modifier = Modifier.fillMaxSize().safeDrawingPadding()) {
    if (hasCamPerm) {
      Box(modifier = Modifier.fillMaxWidth().height(280.dp).clipToBounds()) {
        AndroidView(
          modifier = Modifier.fillMaxSize(),
          factory = { ctx ->
            val previewView = PreviewView(ctx).apply {
              scaleType = PreviewView.ScaleType.FILL_CENTER
            }
            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            val executor = Executors.newSingleThreadExecutor()
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
                              // Pad box slightly so it frames the physical plate instead of slicing through edge letters
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

                              // Same passing event if seen within the last 25 seconds
                              val fullBox = box?.let { Rect(it.left + cropX, it.top + cropY, it.right + cropX, it.bottom + cropY) }
                              val matchIdx = logs.indexOfFirst {
                                now - it.timestampMs < 25_000L && (it.plate == plate || isSameCar(it.plate, plate))
                              }
                              if (matchIdx == -1) {
                                val ts = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault()).format(Date(now))
                                val shotPath = if (debugShots) saveDebugShot(ctx, rotated, fullBox, plate, ts) else ""
                                logs.add(0, ScanEntry(ts, plate, conf, shotPath.ifEmpty { null }, now))
                                rewriteCsv()
                              } else if (conf > logs[matchIdx].confidence && logs[matchIdx].plate != plate) {
                                // Upgrade blurry initial read (e.g. VSO07) to sharper read (VSD07) in-place
                                val old = logs[matchIdx]
                                val shotPath = if (debugShots) saveDebugShot(ctx, rotated, fullBox, plate, old.time) else old.imgPath
                                logs[matchIdx] = old.copy(plate = plate, confidence = conf, imgPath = shotPath, timestampMs = now)
                                rewriteCsv()
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
              provider.unbindAll()
              val cam = provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
              cam.cameraControl.setZoomRatio(1.5f)
            }, ContextCompat.getMainExecutor(ctx))
            previewView
          }
        )
        if (showBoxes && detectedBoxes.isNotEmpty()) {
          Canvas(modifier = Modifier.fillMaxSize()) {
            val (fw, fh) = frameSize
            val sx = size.width / fw
            val sy = size.height / fh
            for (rect in detectedBoxes) {
              drawRect(
                color = Color.Green,
                topLeft = Offset(rect.left * sx, rect.top * sy),
                size = Size(rect.width() * sx, rect.height() * sy),
                style = Stroke(width = 6f)
              )
            }
          }
        }
      }
    } else {
      Text("Vantar leyfi fyrir myndavél", modifier = Modifier.padding(16.dp))
    }

    Column(modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp)) {
      Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
      ) {
        Text("Atvik (${logs.size})", fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
          OutlinedButton(
            enabled = logs.isNotEmpty(),
            onClick = {
              logs.clear()
              logFile.delete()
            }
          ) {
            Text("Hreinsa", maxLines = 1)
          }
          Button(
            enabled = logs.isNotEmpty(),
            onClick = {
              val ordered = if (newestFirst) logs else logs.reversed()
              val csv = "Dagur,Tími,Bílnúmer\n" + ordered.joinToString("\n") { entry ->
                val parts = entry.time.split(" ")
                "${parts.getOrElse(0) { "" }},${parts.getOrElse(1) { entry.time }},${entry.plate}"
              }
              val intent = Intent(Intent.ACTION_SEND).apply {
                type = "text/plain"
                putExtra(Intent.EXTRA_SUBJECT, "Atvik (${logs.size})")
                putExtra(Intent.EXTRA_TEXT, csv)
              }
              context.startActivity(Intent.createChooser(intent, "Deila atvikum"))
            }
          ) {
            Text("Deila", maxLines = 1)
          }
        }
      }
      Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(
          selected = showBoxes,
          onClick = { showBoxes = !showBoxes },
          label = { Text("Rammi") }
        )
        FilterChip(
          selected = debugShots,
          onClick = { debugShots = !debugShots },
          label = { Text("Skjáskot") }
        )
        FilterChip(
          selected = !newestFirst,
          onClick = { newestFirst = !newestFirst },
          label = { Text(if (newestFirst) "Nýjast efst" else "Elst efst") }
        )
      }
    }
    HorizontalDivider()
    val orderedLogs = if (newestFirst) logs else logs.reversed()
    val groupedByDay = orderedLogs.groupBy { it.time.substringBefore(" ") }
    LazyColumn(modifier = Modifier.fillMaxSize().padding(horizontal = 12.dp)) {
      groupedByDay.forEach { (dayStr, dayEntries) ->
        item(key = "day_$dayStr") {
          Surface(
            color = MaterialTheme.colorScheme.surfaceVariant,
            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
          ) {
            Row(
              modifier = Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 6.dp),
              horizontalArrangement = Arrangement.SpaceBetween
            ) {
              Text(formatDayHeader(dayStr), fontWeight = FontWeight.Bold, style = MaterialTheme.typography.bodyMedium)
              Text("${dayEntries.size} atvik", style = MaterialTheme.typography.bodySmall)
            }
          }
        }
        items(dayEntries) { entry ->
          val clockTime = entry.time.substringAfter(" ", entry.time)
          Row(
            modifier = Modifier
              .fillMaxWidth()
              .clickable { selectedEntry = entry }
              .padding(vertical = 8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
          ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(14.dp)) {
              Text(clockTime, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
              Text(entry.plate, fontWeight = FontWeight.Bold, style = MaterialTheme.typography.titleMedium)
            }
            if (entry.imgPath != null && File(entry.imgPath).exists()) {
              val bmp = remember(entry.imgPath) { BitmapFactory.decodeFile(entry.imgPath) }
              if (bmp != null) {
                Image(
                  bitmap = bmp.asImageBitmap(),
                  contentDescription = entry.plate,
                  modifier = Modifier.height(52.dp).width(100.dp)
                )
              }
            }
          }
          HorizontalDivider()
        }
      }
    }

    selectedEntry?.let { entry ->
      AlertDialog(
        onDismissRequest = { selectedEntry = null },
        title = { Text(entry.plate, fontWeight = FontWeight.Bold) },
        text = {
          Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(entry.time)
            val bmp = remember(entry.imgPath) {
              entry.imgPath?.takeIf { File(it).exists() }?.let { BitmapFactory.decodeFile(it) }
            }
            if (bmp != null) {
              Image(
                bitmap = bmp.asImageBitmap(),
                contentDescription = entry.plate,
                modifier = Modifier.fillMaxWidth()
              )
            } else {
              Text("Engin mynd vistuð fyrir þessa færslu.")
            }
          }
        },
        confirmButton = {
          Button(onClick = { shareEntry(context, entry) }) {
            Text("Deila")
          }
        },
        dismissButton = {
          TextButton(onClick = { selectedEntry = null }) {
            Text("Loka")
          }
        }
      )
    }
  }
}
