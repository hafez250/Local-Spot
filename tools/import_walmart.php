<?php
// One-time importer: walmart-products.csv -> OpenCart
ini_set('display_errors', 1);
error_reporting(E_ALL);

$root = __DIR__ . DIRECTORY_SEPARATOR . '..';
$config = $root . DIRECTORY_SEPARATOR . 'config.php';
if (!file_exists($config)) {
    fwrite(STDERR, "config.php not found\n");
    exit(1);
}
require $config;

$csvPath = 'C:/Users/Youssif/Desktop/walmart-products.csv';
if (!file_exists($csvPath)) {
    fwrite(STDERR, "CSV not found: $csvPath\n");
    exit(1);
}

// Connect DB
$mysqli = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($mysqli->connect_error) {
    fwrite(STDERR, "DB connect error: {$mysqli->connect_error}\n");
    exit(1);
}
$mysqli->set_charset('utf8mb4');

// Detect category table columns
$categoryColumns = [];
$resCols = $mysqli->query("SHOW COLUMNS FROM " . DB_PREFIX . "category");
if ($resCols) {
    while ($col = $resCols->fetch_assoc()) {
        $categoryColumns[] = $col['Field'];
    }
}

// Get In Stock status id if exists
$stock_status_id = 7;
$res = $mysqli->query("SELECT stock_status_id FROM " . DB_PREFIX . "stock_status WHERE name='In Stock' LIMIT 1");
if ($res && $row = $res->fetch_assoc()) {
    $stock_status_id = (int)$row['stock_status_id'];
}

$language_id = 1;
$res = $mysqli->query("SELECT language_id FROM " . DB_PREFIX . "language WHERE code='en-gb' LIMIT 1");
if ($res && $row = $res->fetch_assoc()) {
    $language_id = (int)$row['language_id'];
}

// Helpers
function slugify($text) {
    $text = strtolower(trim($text));
    $text = preg_replace('/[^a-z0-9\s-]/', '', $text);
    $text = preg_replace('/\s+/', '-', $text);
    return $text ?: 'category';
}

function getCategoryId($mysqli, $name, $parent_id, $language_id, $categoryColumns) {
    $nameEsc = $mysqli->real_escape_string($name);
    $parent_id = (int)$parent_id;

    $sql = "SELECT c.category_id FROM " . DB_PREFIX . "category c " .
           "JOIN " . DB_PREFIX . "category_description cd ON c.category_id = cd.category_id " .
           "WHERE cd.name='$nameEsc' AND c.parent_id=$parent_id AND cd.language_id=$language_id " .
           "LIMIT 1";
    $res = $mysqli->query($sql);
    if ($res && $row = $res->fetch_assoc()) {
        return (int)$row['category_id'];
    }

    $fields = [];
    $values = [];

    $fields[] = 'parent_id';
    $values[] = (int)$parent_id;

    if (in_array('top', $categoryColumns, true)) {
        $fields[] = '`top`';
        $values[] = 1;
    }
    if (in_array('column', $categoryColumns, true)) {
        $fields[] = '`column`';
        $values[] = 1;
    }
    if (in_array('sort_order', $categoryColumns, true)) {
        $fields[] = 'sort_order';
        $values[] = 0;
    }
    if (in_array('status', $categoryColumns, true)) {
        $fields[] = 'status';
        $values[] = 1;
    }
    if (in_array('date_added', $categoryColumns, true)) {
        $fields[] = 'date_added';
        $values[] = 'NOW()';
    }
    if (in_array('date_modified', $categoryColumns, true)) {
        $fields[] = 'date_modified';
        $values[] = 'NOW()';
    }

    $fieldList = implode(', ', $fields);
    $valueList = implode(', ', array_map(function($v) {
        return is_string($v) && $v !== 'NOW()' ? "'" . $v . "'" : $v;
    }, $values));

    $mysqli->query("INSERT INTO " . DB_PREFIX . "category ($fieldList) VALUES ($valueList)");
    $cat_id = (int)$mysqli->insert_id;

    $mysqli->query("INSERT INTO " . DB_PREFIX . "category_description (category_id, language_id, name, description, meta_title, meta_description, meta_keyword) " .
                    "VALUES ($cat_id, $language_id, '$nameEsc', '', '$nameEsc', '', '')");

    // category_path entries
    if ($parent_id == 0) {
        $mysqli->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($cat_id, $cat_id, 0)");
    } else {
        $res2 = $mysqli->query("SELECT path_id, level FROM " . DB_PREFIX . "category_path WHERE category_id=$parent_id ORDER BY level ASC");
        $level = 0;
        if ($res2) {
            while ($row2 = $res2->fetch_assoc()) {
                $pid = (int)$row2['path_id'];
                $lvl = (int)$row2['level'];
                $mysqli->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($cat_id, $pid, $lvl)");
                $level = $lvl + 1;
            }
        }
        $mysqli->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($cat_id, $cat_id, $level)");
    }

    return $cat_id;
}

function downloadImage($url, $product_id, $root) {
    if (!$url) return '';
    $baseDir = $root . '/image/catalog/walmart';
    if (!is_dir($baseDir)) {
        mkdir($baseDir, 0777, true);
    }
    $ext = pathinfo(parse_url($url, PHP_URL_PATH), PATHINFO_EXTENSION);
    if (!$ext) $ext = 'jpg';
    $fileName = 'p' . $product_id . '.' . $ext;
    $filePath = $baseDir . '/' . $fileName;

    $context = stream_context_create([
        'http' => [
            'timeout' => 1,
            'header'  => "User-Agent: Mozilla/5.0\r\n"
        ],
        'https' => [
            'timeout' => 1,
            'header'  => "User-Agent: Mozilla/5.0\r\n"
        ]
    ]);
    $img = @file_get_contents($url, false, $context);
    if ($img === false) return '';
    file_put_contents($filePath, $img);

    return 'catalog/walmart/' . $fileName;
}

$handle = fopen($csvPath, 'r');
$headers = fgetcsv($handle);
$map = array_flip($headers);

$limit = (int)(getenv('IMPORT_LIMIT') ?: 100);
$start = (int)(getenv('IMPORT_START') ?: 0);
$skipImages = (int)(getenv('IMPORT_SKIP_IMAGES') ?: 0);
$count = 0;
$rowIndex = 0;

while (($row = fgetcsv($handle)) !== false && $count < $limit) {
    $rowIndex++;
    if ($rowIndex <= $start) {
        continue;
    }
    $name = $row[$map['product_name']] ?? '';
    $desc = $row[$map['description']] ?? '';
    $price = $row[$map['final_price']] ?? '';
    $price = is_numeric($price) ? (float)$price : 0.0;
    $sku = $row[$map['sku']] ?? '';
    $main_image = $row[$map['main_image']] ?? '';
    $categoriesJson = $row[$map['categories']] ?? '[]';

    if (!$name) continue;

    // Skip if SKU already exists
    if ($sku) {
        $skuEsc = $mysqli->real_escape_string($sku);
        $resSku = $mysqli->query("SELECT product_id FROM " . DB_PREFIX . "product WHERE sku='$skuEsc' LIMIT 1");
        if ($resSku && $resSku->fetch_assoc()) {
            continue;
        }
    }

    // Insert product
    $nameEsc = $mysqli->real_escape_string($name);
    $descEsc = $mysqli->real_escape_string($desc);
    $skuEsc = $mysqli->real_escape_string((string)$sku);

    $mysqli->query("INSERT INTO " . DB_PREFIX . "product (model, sku, upc, ean, jan, isbn, mpn, location, quantity, stock_status_id, image, manufacturer_id, shipping, price, points, tax_class_id, date_available, weight, weight_class_id, length, width, height, length_class_id, subtract, minimum, sort_order, status, date_added, date_modified) " .
                    "VALUES ('WAL-$count', '$skuEsc', '', '', '', '', '', '', 100, $stock_status_id, '', 0, 1, $price, 0, 0, NOW(), 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, NOW(), NOW())");
    $product_id = (int)$mysqli->insert_id;

    // Download image
    if (!$skipImages) {
        $imagePath = downloadImage(trim($main_image, '\"'), $product_id, $root);
        if ($imagePath) {
            $mysqli->query("UPDATE " . DB_PREFIX . "product SET image='" . $mysqli->real_escape_string($imagePath) . "' WHERE product_id=$product_id");
        }
    }

    // Description
    $mysqli->query("INSERT INTO " . DB_PREFIX . "product_description (product_id, language_id, name, description, tag, meta_title, meta_description, meta_keyword) " .
                    "VALUES ($product_id, $language_id, '$nameEsc', '$descEsc', '', '$nameEsc', '', '')");

    // Store
    $mysqli->query("INSERT INTO " . DB_PREFIX . "product_to_store (product_id, store_id) VALUES ($product_id, 0)");

    // Categories (multi-level)
    $categories = json_decode($categoriesJson, true);
    if (!is_array($categories) || !count($categories)) {
        $categories = [];
    }

    $parent_id = 0;
    foreach ($categories as $catName) {
        if (is_array($catName)) {
            $catName = $catName['name'] ?? '';
        }
        if (!is_string($catName)) {
            continue;
        }
        $catName = trim($catName);
        if (!$catName) continue;
        $cat_id = getCategoryId($mysqli, $catName, $parent_id, $language_id, $categoryColumns);
        $parent_id = $cat_id;
    }

    if ($parent_id > 0) {
        $mysqli->query("INSERT INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $parent_id)");
    }

    $count++;
}

fclose($handle);
$mysqli->close();

echo "Imported $count products\n";
