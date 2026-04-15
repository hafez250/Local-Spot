<?php
$dir = 'c:/xampp/htdocs/opencart1/image/catalog/starnine';
$files = glob($dir . '/*.{jpg,jpeg,png,webp}', GLOB_BRACE);
$minW = PHP_INT_MAX; $minH = PHP_INT_MAX; $maxW = 0; $maxH = 0;
$sample = 0;
foreach ($files as $f) {
  $s = @getimagesize($f);
  if (!$s) continue;
  $w = $s[0]; $h = $s[1];
  $minW = min($minW,$w); $minH = min($minH,$h);
  $maxW = max($maxW,$w); $maxH = max($maxH,$h);
  if ($sample < 5) { echo basename($f) . ": {$w}x{$h}\n"; $sample++; }
}
echo "count=" . count($files) . " min={$minW}x{$minH} max={$maxW}x{$maxH}\n";