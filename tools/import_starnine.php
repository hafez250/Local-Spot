<?php
ini_set('display_errors', 1);
error_reporting(E_ALL);

$root = 'c:/xampp/htdocs/opencart1';
require $root . '/config.php';

$csvPath = 'c:/Users/Youssif/Desktop/starnine/starnine_scraper/data/starninee_products.csv';
if (!file_exists($csvPath)) {
    fwrite(STDERR, "CSV not found\n");
    exit(1);
}

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($m->connect_error) {
    fwrite(STDERR, "DB connect error: {$m->connect_error}\n");
    exit(1);
}
$m->set_charset('utf8mb4');

$language_id = 1;
$res = $m->query("SELECT language_id FROM " . DB_PREFIX . "language WHERE code='en-gb' LIMIT 1");
if ($res && $row = $res->fetch_assoc()) {
    $language_id = (int)$row['language_id'];
}

$stock_status_id = 7;
$res = $m->query("SELECT stock_status_id FROM " . DB_PREFIX . "stock_status WHERE name='In Stock' LIMIT 1");
if ($res && $row = $res->fetch_assoc()) {
    $stock_status_id = (int)$row['stock_status_id'];
}

$categoryColumns = [];
$cols = $m->query("SHOW COLUMNS FROM " . DB_PREFIX . "category");
if ($cols) {
    while ($c = $cols->fetch_assoc()) {
        $categoryColumns[] = $c['Field'];
    }
}

function getOrCreateCategory(mysqli $m, string $name, int $language_id, array $categoryColumns): int {
    $nameEsc = $m->real_escape_string($name);
    $sql = "SELECT c.category_id FROM " . DB_PREFIX . "category c " .
           "JOIN " . DB_PREFIX . "category_description cd ON cd.category_id = c.category_id " .
           "WHERE c.parent_id=0 AND cd.language_id=$language_id AND cd.name='$nameEsc' LIMIT 1";
    $r = $m->query($sql);
    if ($r && $row = $r->fetch_assoc()) {
        return (int)$row['category_id'];
    }

    $fields = ['parent_id'];
    $values = ['0'];

    if (in_array('top', $categoryColumns, true)) { $fields[] = '`top`'; $values[] = '1'; }
    if (in_array('column', $categoryColumns, true)) { $fields[] = '`column`'; $values[] = '1'; }
    if (in_array('sort_order', $categoryColumns, true)) { $fields[] = 'sort_order'; $values[] = '0'; }
    if (in_array('status', $categoryColumns, true)) { $fields[] = 'status'; $values[] = '1'; }
    if (in_array('date_added', $categoryColumns, true)) { $fields[] = 'date_added'; $values[] = 'NOW()'; }
    if (in_array('date_modified', $categoryColumns, true)) { $fields[] = 'date_modified'; $values[] = 'NOW()'; }

    $m->query("INSERT INTO " . DB_PREFIX . "category (" . implode(',', $fields) . ") VALUES (" . implode(',', $values) . ")");
    $cat_id = (int)$m->insert_id;

    $m->query("INSERT INTO " . DB_PREFIX . "category_description (category_id, language_id, name, description, meta_title, meta_description, meta_keyword) " .
              "VALUES ($cat_id, $language_id, '$nameEsc', '', '$nameEsc', '', '')");

    $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($cat_id, $cat_id, 0)");
    return $cat_id;
}

function downloadImage(string $url, string $sku, string $root): string {
    if (!$url) return '';
    $dir = $root . '/image/catalog/starnine';
    if (!is_dir($dir)) {
        mkdir($dir, 0777, true);
    }
    $path = parse_url($url, PHP_URL_PATH) ?: '';
    $ext = pathinfo($path, PATHINFO_EXTENSION);
    if (!$ext) $ext = 'jpg';

    $safeSku = preg_replace('/[^a-zA-Z0-9_-]/', '-', $sku);
    $filename = strtolower($safeSku) . '.' . strtolower($ext);
    $full = $dir . '/' . $filename;

    if (!file_exists($full)) {
        $ctx = stream_context_create([
            'http' => ['timeout' => 8, 'header' => "User-Agent: Mozilla/5.0\r\n"],
            'https' => ['timeout' => 8, 'header' => "User-Agent: Mozilla/5.0\r\n"]
        ]);
        $img = @file_get_contents($url, false, $ctx);
        if ($img !== false) {
            file_put_contents($full, $img);
        }
    }

    return file_exists($full) ? ('catalog/starnine/' . $filename) : '';
}

$category_id = getOrCreateCategory($m, 'Starnine 2026', $language_id, $categoryColumns);

$h = fopen($csvPath, 'r');
$headers = fgetcsv($h);
$map = array_flip($headers);

$inserted = 0;
$updated = 0;
$skipped = 0;

while (($row = fgetcsv($h)) !== false) {
    $title = trim($row[$map['Title']] ?? '');
    $handle = trim($row[$map['Handle']] ?? '');
    $categoryName = trim($row[$map['Category']] ?? '');
    $priceRaw = trim($row[$map['Price']] ?? '0');
    $description = trim($row[$map['Description']] ?? '');
    $variants = trim($row[$map['Variants']] ?? '');
    $imageUrl = trim($row[$map['Image_URL']] ?? '');

    if ($title === '' || $handle === '') {
        $skipped++;
        continue;
    }

    $skuEsc = $m->real_escape_string($handle);
    $exists = $m->query("SELECT product_id FROM " . DB_PREFIX . "product WHERE sku='$skuEsc' LIMIT 1");
    $existingRow = ($exists && $exists->num_rows) ? $exists->fetch_assoc() : null;

    $price = is_numeric($priceRaw) ? (float)$priceRaw : 0.0;
    $titleEsc = $m->real_escape_string($title);
    $modelEsc = $m->real_escape_string($title);
    $descText = $description !== '' ? $description : 'No description';
    if ($variants !== '') {
        $descText .= "\nVariants: " . $variants;
    }
    $descText .= "\nHandle: " . $handle;
    $descEsc = $m->real_escape_string($descText);

    $imagePath = downloadImage($imageUrl, $handle, $root);
    $imageEsc = $m->real_escape_string($imagePath);

    $rowCategoryId = $category_id;
    if ($categoryName !== '') {
        $rowCategoryId = getOrCreateCategory($m, $categoryName, $language_id, $categoryColumns);
    }

    if ($existingRow) {
        $product_id = (int)$existingRow['product_id'];
        $m->query("UPDATE " . DB_PREFIX . "product SET model='$modelEsc', price=$price, quantity=100, stock_status_id=$stock_status_id, status=1, image='$imageEsc', date_modified=NOW() WHERE product_id=$product_id");
        $m->query("UPDATE " . DB_PREFIX . "product_description SET name='$titleEsc', description='$descEsc', meta_title='$titleEsc' WHERE product_id=$product_id AND language_id=$language_id");

        $existsStore = $m->query("SELECT 1 FROM " . DB_PREFIX . "product_to_store WHERE product_id=$product_id AND store_id=0");
        if (!$existsStore || !$existsStore->num_rows) {
            $m->query("INSERT INTO " . DB_PREFIX . "product_to_store (product_id, store_id) VALUES ($product_id, 0)");
        }

        $existsCategory = $m->query("SELECT 1 FROM " . DB_PREFIX . "product_to_category WHERE product_id=$product_id AND category_id=$rowCategoryId");
        if (!$existsCategory || !$existsCategory->num_rows) {
            $m->query("INSERT INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $rowCategoryId)");
        }

        $updated++;
    } else {
        $m->query("INSERT INTO " . DB_PREFIX . "product (model, sku, upc, ean, jan, isbn, mpn, location, quantity, stock_status_id, image, manufacturer_id, shipping, price, points, tax_class_id, date_available, weight, weight_class_id, length, width, height, length_class_id, subtract, minimum, sort_order, status, date_added, date_modified) " .
                  "VALUES ('$modelEsc', '$skuEsc', '', '', '', '', '', '', 100, $stock_status_id, '$imageEsc', 0, 1, $price, 0, 0, NOW(), 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, NOW(), NOW())");
        $product_id = (int)$m->insert_id;

        $m->query("INSERT INTO " . DB_PREFIX . "product_description (product_id, language_id, name, description, tag, meta_title, meta_description, meta_keyword) " .
                  "VALUES ($product_id, $language_id, '$titleEsc', '$descEsc', '', '$titleEsc', '', '')");
        $m->query("INSERT INTO " . DB_PREFIX . "product_to_store (product_id, store_id) VALUES ($product_id, 0)");
        $m->query("INSERT INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $rowCategoryId)");

        $inserted++;
    }
}

fclose($h);
$m->close();

echo "Inserted=$inserted Updated=$updated Skipped=$skipped CategoryID=$category_id" . PHP_EOL;
