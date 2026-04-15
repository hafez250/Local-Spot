<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

// Helpers
function tableExists(mysqli $m, string $table): bool {
  $r = $m->query("SHOW TABLES LIKE '" . $m->real_escape_string($table) . "'");
  return $r && $r->num_rows > 0;
}

function colExists(mysqli $m, string $table, string $col): bool {
  $r = $m->query("SHOW COLUMNS FROM `" . $table . "` LIKE '" . $m->real_escape_string($col) . "'");
  return $r && $r->num_rows > 0;
}

$language_id = 1;
$res = $m->query("SELECT language_id FROM " . DB_PREFIX . "language WHERE code='en-gb' LIMIT 1");
if ($res && $row = $res->fetch_assoc()) {
  $language_id = (int)$row['language_id'];
}

$categoryTable = DB_PREFIX . 'category';
$categoryDescTable = DB_PREFIX . 'category_description';
$categoryPathTable = DB_PREFIX . 'category_path';
$categoryStoreTable = DB_PREFIX . 'category_to_store';

function getOrCreateTopCategory(mysqli $m, int $language_id, string $name): int {
  $nameEsc = $m->real_escape_string($name);
  $q = $m->query("SELECT c.category_id FROM " . DB_PREFIX . "category c JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=$language_id WHERE c.parent_id=0 AND cd.name='$nameEsc' LIMIT 1");
  if ($q && $q->num_rows) {
    $r = $q->fetch_assoc();
    return (int)$r['category_id'];
  }

  $fields = ['parent_id'];
  $values = ['0'];

  if (colExists($m, DB_PREFIX . 'category', 'top')) { $fields[] = '`top`'; $values[] = '1'; }
  if (colExists($m, DB_PREFIX . 'category', 'column')) { $fields[] = '`column`'; $values[] = '1'; }
  if (colExists($m, DB_PREFIX . 'category', 'sort_order')) { $fields[] = 'sort_order'; $values[] = '0'; }
  if (colExists($m, DB_PREFIX . 'category', 'status')) { $fields[] = 'status'; $values[] = '1'; }
  if (colExists($m, DB_PREFIX . 'category', 'date_added')) { $fields[] = 'date_added'; $values[] = 'NOW()'; }
  if (colExists($m, DB_PREFIX . 'category', 'date_modified')) { $fields[] = 'date_modified'; $values[] = 'NOW()'; }

  $m->query("INSERT INTO " . DB_PREFIX . "category (" . implode(',', $fields) . ") VALUES (" . implode(',', $values) . ")");
  $category_id = (int)$m->insert_id;

  $m->query("INSERT INTO " . DB_PREFIX . "category_description (category_id, language_id, name, description, meta_title, meta_description, meta_keyword) VALUES ($category_id, $language_id, '$nameEsc', '', '$nameEsc', '', '')");

  if (tableExists($m, DB_PREFIX . 'category_to_store')) {
    $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($category_id, 0)");
  }

  if (tableExists($m, DB_PREFIX . 'category_path')) {
    $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($category_id, $category_id, 0)");
  }

  return $category_id;
}

function getCategoryByName(mysqli $m, int $language_id, string $name): ?int {
  $nameEsc = $m->real_escape_string($name);
  $q = $m->query("SELECT cd.category_id FROM " . DB_PREFIX . "category_description cd WHERE cd.language_id=$language_id AND cd.name='$nameEsc' LIMIT 1");
  if ($q && $q->num_rows) {
    $r = $q->fetch_assoc();
    return (int)$r['category_id'];
  }
  return null;
}

function rebuildPath(mysqli $m, int $category_id, int $parent_id): void {
  if (!tableExists($m, DB_PREFIX . 'category_path')) return;

  $m->query("DELETE FROM " . DB_PREFIX . "category_path WHERE category_id=$category_id");

  $level = 0;
  $parentPath = $m->query("SELECT path_id, level FROM " . DB_PREFIX . "category_path WHERE category_id=$parent_id ORDER BY level ASC");
  if ($parentPath && $parentPath->num_rows) {
    while ($row = $parentPath->fetch_assoc()) {
      $pid = (int)$row['path_id'];
      $lvl = (int)$row['level'];
      $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($category_id, $pid, $lvl)");
      $level = $lvl + 1;
    }
  }

  $m->query("INSERT INTO " . DB_PREFIX . "category_path (category_id, path_id, level) VALUES ($category_id, $category_id, $level)");
}

$clothing_id = getOrCreateTopCategory($m, $language_id, 'Clothing');
$svg_id = getCategoryByName($m, $language_id, 'Svg');
$starnine_id = getCategoryByName($m, $language_id, 'Starnine 2026');

if (!$svg_id && !$starnine_id) {
  echo "No Svg or Starnine category found" . PHP_EOL;
  exit;
}

$m->begin_transaction();
try {
  foreach ([$svg_id, $starnine_id] as $bid) {
    if (!$bid) continue;

    // Move under Clothing
    $set = ["parent_id=$clothing_id", "status=1"];
    if (colExists($m, DB_PREFIX . 'category', 'date_modified')) {
      $set[] = "date_modified=NOW()";
    }
    $m->query("UPDATE " . DB_PREFIX . "category SET " . implode(', ', $set) . " WHERE category_id=$bid");

    if (colExists($m, DB_PREFIX . 'category', 'top')) {
      $m->query("UPDATE " . DB_PREFIX . "category SET `top`=0 WHERE category_id=$bid");
    }

    // Ensure store mapping exists
    if (tableExists($m, DB_PREFIX . 'category_to_store')) {
      $check = $m->query("SELECT 1 FROM " . DB_PREFIX . "category_to_store WHERE category_id=$bid AND store_id=0 LIMIT 1");
      if (!$check || !$check->num_rows) {
        $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($bid, 0)");
      }
    }

    rebuildPath($m, $bid, $clothing_id);
  }

  // Ensure clothing itself is store-mapped and top-level path exists
  if (tableExists($m, DB_PREFIX . 'category_to_store')) {
    $check = $m->query("SELECT 1 FROM " . DB_PREFIX . "category_to_store WHERE category_id=$clothing_id AND store_id=0 LIMIT 1");
    if (!$check || !$check->num_rows) {
      $m->query("INSERT INTO " . DB_PREFIX . "category_to_store (category_id, store_id) VALUES ($clothing_id, 0)");
    }
  }

  $m->commit();
} catch (Throwable $e) {
  $m->rollback();
  throw $e;
}

// Verify
$verify = $m->query("SELECT c.category_id, c.parent_id, cd.name FROM " . DB_PREFIX . "category c JOIN " . DB_PREFIX . "category_description cd ON cd.category_id=c.category_id AND cd.language_id=$language_id WHERE c.category_id IN ($clothing_id" . ($svg_id ? ",$svg_id" : "") . ($starnine_id ? ",$starnine_id" : "") . ") ORDER BY c.category_id");
while ($row = $verify->fetch_assoc()) {
  echo $row['category_id'] . " parent=" . $row['parent_id'] . " name=" . $row['name'] . PHP_EOL;
}
