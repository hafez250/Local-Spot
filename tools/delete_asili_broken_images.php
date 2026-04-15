<?php
ini_set('display_errors', 1);
error_reporting(E_ALL);

require 'c:/xampp/htdocs/opencart1/config.php';

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($m->connect_error) {
    fwrite(STDERR, "DB connect error: {$m->connect_error}" . PHP_EOL);
    exit(1);
}
$m->set_charset('utf8mb4');

$language_id = 1;
$res = $m->query("SELECT language_id FROM " . DB_PREFIX . "language WHERE code='en-gb' LIMIT 1");
if ($res && ($row = $res->fetch_assoc())) {
    $language_id = (int)$row['language_id'];
}

$asili_id = 0;
$idQuery = $m->query(
    "SELECT c.category_id
     FROM " . DB_PREFIX . "category c
     JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id
     WHERE c.parent_id=492 AND cd.language_id=$language_id AND LOWER(cd.name) LIKE 'asili%'
     ORDER BY c.category_id DESC
     LIMIT 1"
);
if ($idQuery && ($idRow = $idQuery->fetch_assoc())) {
    $asili_id = (int)$idRow['category_id'];
}

if ($asili_id <= 0) {
    echo "Asili category not found under Clothing (492)." . PHP_EOL;
    exit(1);
}

$productIds = [];
$productQuery = $m->query(
    "SELECT DISTINCT p.product_id, p.image
     FROM " . DB_PREFIX . "product p
     JOIN " . DB_PREFIX . "product_to_category p2c ON p2c.product_id = p.product_id
     JOIN " . DB_PREFIX . "category_path cp ON cp.category_id = p2c.category_id
     WHERE cp.path_id = $asili_id"
);

$brokenIds = [];
$checked = 0;

if ($productQuery) {
    while ($row = $productQuery->fetch_assoc()) {
        $checked++;
        $product_id = (int)$row['product_id'];
        $image = trim((string)$row['image']);
        $isBroken = false;

        $imageLower = strtolower(str_replace('\\', '/', $image));

        $ext = strtolower((string)pathinfo($imageLower, PATHINFO_EXTENSION));

        if ($image === '') {
            $isBroken = true;
        } elseif ($imageLower === 'placeholder.png' || $imageLower === 'no_image.png') {
            // Treat system placeholder images as "no real image".
            $isBroken = true;
        } elseif (in_array($ext, ['heic', 'heif'], true)) {
            // HEIC/HEIF do not reliably render in current product card flow.
            $isBroken = true;
        } else {
            $fullPath = DIR_IMAGE . html_entity_decode($image, ENT_QUOTES, 'UTF-8');

            if (!is_file($fullPath)) {
                $isBroken = true;
            } else {
                $info = @getimagesize($fullPath);
                if ($info === false) {
                    $isBroken = true;
                }
            }
        }

        if ($isBroken) {
            $brokenIds[] = $product_id;
        }
    }
}

$brokenIds = array_values(array_unique(array_map('intval', $brokenIds)));

if (!$brokenIds) {
    echo "AsiliCategoryID=$asili_id Checked=$checked Broken=0 Deleted=0" . PHP_EOL;
    exit(0);
}

$idList = implode(',', $brokenIds);

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
    'review',
    'cart',
    'wishlist',
    'customer_wishlist'
];

foreach ($tables as $t) {
    $table = DB_PREFIX . $t;
    $check = $m->query("SHOW TABLES LIKE '" . $m->real_escape_string($table) . "'");

    if ($check && $check->num_rows) {
        $m->query("DELETE FROM `$table` WHERE product_id IN ($idList)");
    }
}

$remainingRes = $m->query(
    "SELECT COUNT(DISTINCT p2c.product_id) AS c
     FROM " . DB_PREFIX . "product_to_category p2c
     JOIN " . DB_PREFIX . "category_path cp ON cp.category_id = p2c.category_id
     WHERE cp.path_id = $asili_id"
);
$remaining = 0;
if ($remainingRes && ($r = $remainingRes->fetch_assoc())) {
    $remaining = (int)$r['c'];
}

echo "AsiliCategoryID=$asili_id Checked=$checked Broken=" . count($brokenIds) . " Deleted=" . count($brokenIds) . " Remaining=$remaining" . PHP_EOL;
