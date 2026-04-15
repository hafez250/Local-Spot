<?php
ini_set('display_errors', 1);
error_reporting(E_ALL);

$root = 'c:/xampp/htdocs/opencart1';
require $root . '/config.php';

$csvPath = 'c:/Users/Youssif/Desktop/starnine/starnine_scraper/data/svg_categorized.csv';
if (!file_exists($csvPath)) {
    fwrite(STDERR, "CSV not found: $csvPath" . PHP_EOL);
    exit(1);
}

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
if ($m->connect_error) {
    fwrite(STDERR, "DB connect error: {$m->connect_error}" . PHP_EOL);
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

$clothing_id = 492;
$svg_id = 62;

function ensureCategoryColumns(mysqli $m): array {
    $out = [];
    $q = $m->query("SHOW COLUMNS FROM " . DB_PREFIX . "category");
    if ($q) {
        while ($r = $q->fetch_assoc()) {
            $out[] = $r['Field'];
        }
    }
    return $out;
}

function getCategoryByNameAndParent(mysqli $m, int $language_id, string $name, int $parent_id): ?int {
    $nameEsc = $m->real_escape_string($name);
    $q = $m->query(
        "SELECT c.category_id FROM " . DB_PREFIX . "category c
         JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id
         WHERE cd.language_id=$language_id AND cd.name='$nameEsc' AND c.parent_id=$parent_id
         LIMIT 1"
    );
    if ($q && ($r = $q->fetch_assoc())) {
        return (int)$r['category_id'];
    }
    return null;
}

function createChildCategory(mysqli $m, int $language_id, string $name, int $parent_id, array $categoryColumns): int {
    $nameEsc = $m->real_escape_string($name);

    $fields = ['parent_id'];
    $values = [(string)$parent_id];
    if (in_array('top', $categoryColumns, true)) { $fields[] = '`top`'; $values[] = '0'; }
    if (in_array('column', $categoryColumns, true)) { $fields[] = '`column`'; $values[] = '1'; }
    if (in_array('sort_order', $categoryColumns, true)) { $fields[] = 'sort_order'; $values[] = '0'; }
    if (in_array('status', $categoryColumns, true)) { $fields[] = 'status'; $values[] = '1'; }
    if (in_array('date_added', $categoryColumns, true)) { $fields[] = 'date_added'; $values[] = 'NOW()'; }
    if (in_array('date_modified', $categoryColumns, true)) { $fields[] = 'date_modified'; $values[] = 'NOW()'; }

    $m->query("INSERT INTO " . DB_PREFIX . "category (" . implode(',', $fields) . ") VALUES (" . implode(',', $values) . ")");
    $category_id = (int)$m->insert_id;

    $m->query("INSERT INTO " . DB_PREFIX . "category_description (category_id, language_id, name, description, meta_title, meta_description, meta_keyword)
               VALUES ($category_id, $language_id, '$nameEsc', '', '$nameEsc', '', '')");

    $storeCheck = $m->query("SELECT 1 FROM " . DB_PREFIX . "category_to_store WHERE category_id=$category_id AND store_id=0 LIMIT 1");
    if (!$storeCheck || !$storeCheck->num_rows) {
        $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($category_id, 0)");
    }

    $m->query("DELETE FROM " . DB_PREFIX . "category_path WHERE category_id=$category_id");
    $parentPath = $m->query("SELECT path_id, level FROM " . DB_PREFIX . "category_path WHERE category_id=$parent_id ORDER BY level ASC");
    $level = 0;
    if ($parentPath) {
        while ($pr = $parentPath->fetch_assoc()) {
            $pid = (int)$pr['path_id'];
            $lvl = (int)$pr['level'];
            $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($category_id, $pid, $lvl)");
            $level = $lvl + 1;
        }
    }
    $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($category_id, $category_id, $level)");

    return $category_id;
}

function normalizeCategoryLabel(string $label): string {
    $label = trim($label);
    $label = preg_replace('/\s+/', ' ', $label);
    return $label;
}

function parseCategoryChain(string $raw): array {
    $raw = trim($raw);
    if ($raw === '' || $raw === '[]') {
        return [];
    }

    $parts = preg_split('/\s*>\s*|\s*\/\s*/', $raw);
    $out = [];

    foreach ($parts as $part) {
        $name = normalizeCategoryLabel((string)$part);
        if ($name !== '') {
            $out[] = $name;
        }
    }

    return $out;
}

function getOrCreateNestedCategory(
    mysqli $m,
    int $language_id,
    int $root_parent_id,
    array $categoryColumns,
    array $chain
): int {
    $parent = $root_parent_id;

    foreach ($chain as $name) {
        $existing = getCategoryByNameAndParent($m, $language_id, $name, $parent);

        if ($existing) {
            $m->query("UPDATE " . DB_PREFIX . "category SET status=1, parent_id=$parent WHERE category_id=$existing");

            $storeCheck = $m->query("SELECT 1 FROM " . DB_PREFIX . "category_to_store WHERE category_id=$existing AND store_id=0 LIMIT 1");
            if (!$storeCheck || !$storeCheck->num_rows) {
                $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($existing, 0)");
            }

            $parent = $existing;
        } else {
            $parent = createChildCategory($m, $language_id, $name, $parent, $categoryColumns);
        }
    }

    return $parent;
}

function downloadImage(string $url, string $sku, string $root): string {
    if (!$url) return '';
    $dir = $root . '/image/catalog/svg';
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
            'http' => ['timeout' => 12, 'header' => "User-Agent: Mozilla/5.0\r\n"],
            'https' => ['timeout' => 12, 'header' => "User-Agent: Mozilla/5.0\r\n"]
        ]);
        $img = @file_get_contents($url, false, $ctx);
        if ($img !== false) {
            file_put_contents($full, $img);
        }
    }

    return file_exists($full) ? ('catalog/svg/' . $filename) : '';
}

$categoryColumns = ensureCategoryColumns($m);

$h = fopen($csvPath, 'r');
$headers = fgetcsv($h);
if (!$headers) {
    fwrite(STDERR, "CSV header missing." . PHP_EOL);
    exit(1);
}
$map = array_flip($headers);

$inserted = 0;
$updated = 0;
$skipped = 0;

while (($row = fgetcsv($h)) !== false) {
    $rawCategory = trim($row[$map['Category']] ?? '');
    $title = trim($row[$map['Title']] ?? '');
    $priceRaw = trim($row[$map['Price']] ?? '0');
    $description = trim($row[$map['Description']] ?? '');
    $variants = trim($row[$map['Variants']] ?? '');
    $handle = trim($row[$map['Handle']] ?? '');
    $imageUrl = trim($row[$map['Image_URL']] ?? '');

    if ($title === '' || $handle === '') {
        $skipped++;
        continue;
    }

    $categoryChain = parseCategoryChain($rawCategory);
    $targetCategoryId = $svg_id;
    if ($categoryChain) {
        $targetCategoryId = getOrCreateNestedCategory($m, $language_id, $svg_id, $categoryColumns, $categoryChain);
    }

    $price = is_numeric($priceRaw) ? (float)$priceRaw : 0.0;
    $titleEsc = $m->real_escape_string($title);
    $modelEsc = $m->real_escape_string($title);
    $skuKey = mb_substr($handle, 0, 64, 'UTF-8');
    $skuEsc = $m->real_escape_string($skuKey);

    $descText = ($description !== '' ? $description : 'No description');
    if ($variants !== '') {
        $descText .= "\nVariants: " . $variants;
    }
    if ($rawCategory !== '') {
        $descText .= "\nCategory: " . $rawCategory;
    }
    $descText .= "\nHandle: " . $handle;
    $descEsc = $m->real_escape_string($descText);

    $imagePath = downloadImage($imageUrl, $skuKey, $root);
    $imageEsc = $m->real_escape_string($imagePath);

    $exists = $m->query("SELECT product_id FROM " . DB_PREFIX . "product WHERE sku='$skuEsc' LIMIT 1");
    $existing = ($exists && $exists->num_rows) ? $exists->fetch_assoc() : null;

    if ($existing) {
        $product_id = (int)$existing['product_id'];
        $m->query("UPDATE " . DB_PREFIX . "product
          SET model='$modelEsc', price=$price, quantity=100, stock_status_id=$stock_status_id, status=1, image='$imageEsc', date_modified=NOW()
          WHERE product_id=$product_id");
        $m->query("UPDATE " . DB_PREFIX . "product_description
          SET name='$titleEsc', description='$descEsc', meta_title='$titleEsc'
          WHERE product_id=$product_id AND language_id=$language_id");

        $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_store (product_id, store_id) VALUES ($product_id, 0)");
        $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $svg_id)");
        $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $clothing_id)");
        if ($targetCategoryId !== $svg_id) {
            $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $targetCategoryId)");
        }

        // Keep one direct Svg child assignment per product.
        $childQuery = $m->query("SELECT category_id FROM " . DB_PREFIX . "category WHERE parent_id=$svg_id");
        $childIds = [];
        if ($childQuery) {
            while ($cr = $childQuery->fetch_assoc()) {
                $cid = (int)$cr['category_id'];
                if ($cid !== $targetCategoryId) {
                    $childIds[] = $cid;
                }
            }
        }
        if ($childIds) {
            $m->query("DELETE FROM " . DB_PREFIX . "product_to_category WHERE product_id=$product_id AND category_id IN (" . implode(',', $childIds) . ")");
        }

        $updated++;
    } else {
        $m->query("INSERT INTO " . DB_PREFIX . "product
          (model, sku, upc, ean, jan, isbn, mpn, location, quantity, stock_status_id, image, manufacturer_id, shipping, price, points, tax_class_id, date_available, weight, weight_class_id, length, width, height, length_class_id, subtract, minimum, sort_order, status, date_added, date_modified)
          VALUES
          ('$modelEsc', '$skuEsc', '', '', '', '', '', '', 100, $stock_status_id, '$imageEsc', 0, 1, $price, 0, 0, NOW(), 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, NOW(), NOW())");
        $product_id = (int)$m->insert_id;

        $m->query("INSERT INTO " . DB_PREFIX . "product_description
          (product_id, language_id, name, description, tag, meta_title, meta_description, meta_keyword)
          VALUES
          ($product_id, $language_id, '$titleEsc', '$descEsc', '', '$titleEsc', '', '')");

        $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_store (product_id, store_id) VALUES ($product_id, 0)");
        $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $svg_id)");
        $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $clothing_id)");
        if ($targetCategoryId !== $svg_id) {
            $m->query("INSERT IGNORE INTO " . DB_PREFIX . "product_to_category (product_id, category_id) VALUES ($product_id, $targetCategoryId)");
        }

        $inserted++;
    }
}

fclose($h);

$verify = $m->query("SELECT COUNT(DISTINCT product_id) AS c FROM " . DB_PREFIX . "product_to_category WHERE category_id=$svg_id");
$count = ($verify && ($r = $verify->fetch_assoc())) ? (int)$r['c'] : 0;

$m->close();

echo "Inserted=$inserted Updated=$updated Skipped=$skipped SvgCategoryID=$svg_id SvgProducts=$count" . PHP_EOL;

