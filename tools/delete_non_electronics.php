<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$keep = [
  'Electronics','iPhone','Iphone','iPad','MacBook','iMac','Apple Watch','AirPods'
];
$keepEsc = array_map(fn($v) => "'" . $m->real_escape_string($v) . "'", $keep);
$keepList = implode(',', $keepEsc);

// products to keep
$keepSql = "SELECT DISTINCT p.product_id FROM " . DB_PREFIX . "product p
JOIN " . DB_PREFIX . "product_to_category p2c ON p.product_id = p2c.product_id
JOIN " . DB_PREFIX . "category_description cd ON cd.category_id = p2c.category_id
WHERE cd.language_id = 1 AND cd.name IN ($keepList)";
$keepRes = $m->query($keepSql);
$keepIds = [];
while ($row = $keepRes->fetch_assoc()) { $keepIds[] = (int)$row['product_id']; }

$idsSql = '';
if (count($keepIds)) {
  $idsSql = implode(',', $keepIds);
  $whereDelete = "product_id NOT IN ($idsSql)";
} else {
  $whereDelete = "1=1";
}

$delRes = $m->query("SELECT product_id FROM " . DB_PREFIX . "product WHERE $whereDelete");
$delIds = [];
while ($r = $delRes->fetch_assoc()) { $delIds[] = (int)$r['product_id']; }

if (!count($delIds)) {
  echo "No products deleted.\n";
  exit;
}

$delList = implode(',', $delIds);
$tables = [
  'product',
  'product_description',
  'product_attribute',
  'product_discount',
  'product_filter',
  'product_image',
  'product_option',
  'product_option_value',
  'product_related',
  'product_reward',
  'product_special',
  'product_to_category',
  'product_to_download',
  'product_to_layout',
  'product_to_store',
  'product_recurring',
  'product_subscription',
  'wishlist',
  'cart'
];

foreach ($tables as $t) {
  $check = $m->query("SHOW TABLES LIKE '" . DB_PREFIX . $m->real_escape_string($t) . "'");
  if ($check && $check->num_rows) {
    $m->query("DELETE FROM " . DB_PREFIX . "$t WHERE product_id IN ($delList)");
  }
}

$r1 = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "product");
$r2 = $m->query("SELECT COUNT(*) c FROM " . DB_PREFIX . "product WHERE product_id IN (" . (count($keepIds)?$idsSql:'0') . ")");
$total = $r1->fetch_assoc()['c'];
$kept = $r2->fetch_assoc()['c'];

echo "Deleted=" . count($delIds) . " Kept=" . $kept . " TotalNow=" . $total . PHP_EOL;
