<?php
// One-time importer: apple_products_with_images.csv -> OpenCart
ini_set('display_errors', 1);
error_reporting(E_ALL);

$root = __DIR__ . DIRECTORY_SEPARATOR . '..';
$config = $root . DIRECTORY_SEPARATOR . 'config.php';
if (!file_exists($config)) {
    fwrite(STDERR, "config.php not found\n");
    exit(1);
}
require $config;

$csvPath = 'C:/xampp/htdocs/opencart1/apple_products_with_images.csv';
if (!file_exists($csvPath)) {
    fwrite(STDERR, "CSV not found: $csvPath\n");
    exit(1);
}

$mysqli = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($mysqli->connect_error) {
    fwrite(STDERR, "DB connect error: {$mysqli->connect_error}\n");
    exit(1);
}
$mysqli->set_charset('utf8mb4');

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

$categoryColumns = [];
$resCols = $mysqli->query("SHOW COLUMNS FROM " . DB_PREFIX . "category");
if ($resCols) {
    while ($col = $resCols->fetch_assoc()) {
        $categoryColumns[] = $col['Field'];
    }
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

$handle = fopen($csvPath, 'r');
$headers = fgetcsv($handle);
$map = array_flip($headers);

$limit = 100;
$count = 0;

while (($row = fgetcsv($handle)) !== false && $count < $limit) {
    $product_id = $row[$map['product_id']] ?? '';
    $category = $row[$map['category']] ?? '';
    $model = $row[$map['model_name']] ?? '';
    $price = $row[$map['price']] ?? '';
    $price = is_numeric($price) ? (float)$price : 0.0;
    $desc = 'Model: ' . ($row[$map['model_name']] ?? '') . "\n" .
            'Release Year: ' . ($row[$map['release_year']] ?? '') . "\n" .
            'Color: ' . ($row[$map['color']] ?? '') . "\n" .
            'Storage: ' . ($row[$map['storage']] ?? '') . "\n" .
            'RAM: ' . ($row[$map['ram']] ?? '') . "\n" .
            'CPU: ' . ($row[$map['cpu']] ?? '') . "\n" .
            'GPU: ' . ($row[$map['gpu']] ?? '') . "\n" .
            'Screen Size: ' . ($row[$map['screen_size']] ?? '') . "\n" .
            'Battery: ' . ($row[$map['battery_mAh']] ?? '') . "\n" .
            'Weight: ' . ($row[$map['weight_grams']] ?? '') . "\n" .
            'Dimensions: ' . ($row[$map['dimensions_mm']] ?? '') . "\n" .
            'Warranty: ' . ($row[$map['warranty']] ?? '') . "\n" .
            'Country: ' . ($row[$map['country_origin']] ?? '') . "\n";
    $name = $row[$map['model_name']] ?? '';
    $image_path = $row[$map['image_path']] ?? '';
    $in_stock = ($row[$map['in_stock']] ?? 'True') === 'True' ? 1 : 0;

    if (!$name) continue;

    $nameEsc = $mysqli->real_escape_string($name);
    $descEsc = $mysqli->real_escape_string($desc);
    $skuEsc = $mysqli->real_escape_string((string)$product_id);

    $resSku = $mysqli->query("SELECT product_id FROM " . DB_PREFIX . "product WHERE sku='$skuEsc' LIMIT 1");
    if ($resSku && $resSku->fetch_assoc()) {
        continue;
    }

    $mysqli->query("INSERT INTO " . DB_PREFIX . "product (model, sku, upc, ean, jan, isbn, mpn, location, quantity, stock_status_id, image, manufacturer_id, shipping, price, points, tax_class_id, date_available, weight, weight_class_id, length, width, height, length_class_id, subtract, minimum, sort_order, status, date_added, date_modified) " .
                    "VALUES ('$model', '$skuEsc', '', '', '', '', '', '', 100, $stock_status_id, '$image_path', 0, 1, $price, 0, 0, NOW(), 0, 0, 0, 0, 0, 0, 1, 1, 0, $in_stock, NOW(), NOW())");
    $new_product_id = (int)$mysqli->insert_id;

    $mysqli->query("INSERT INTO " . DB_PREFIX . "product_description (product_id, language_id, name, description, tag, meta_title, meta_description, meta_keyword) " .
                    "VALUES ($new_product_id, $language_id, '$nameEsc', '$descEsc', '', '$nameEsc', '', '')");

    $mysqli->query("INSERT INTO " . DB_PREFIX . "product_to_store (product_id, store_id) VALUES ($new_product_id, 0)");

    if ($category) {
        $cat_id = getCategoryId($mysqli, $category, 0, $language_id, $categoryColumns);
        $mysqli->query("INSERT INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($new_product_id, $cat_id)");
    }

    $count++;
}

fclose($handle);
$mysqli->close();

echo "Imported $count products\n";