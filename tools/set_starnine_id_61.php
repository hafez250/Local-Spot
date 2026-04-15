<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$oldId = 490;
$newId = 61;

function tableExists(mysqli $m, string $table): bool {
  $r = $m->query("SHOW TABLES LIKE '" . $m->real_escape_string($table) . "'");
  return $r && $r->num_rows > 0;
}

function colExists(mysqli $m, string $table, string $col): bool {
  $r = $m->query("SHOW COLUMNS FROM `" . $table . "` LIKE '" . $m->real_escape_string($col) . "'");
  return $r && $r->num_rows > 0;
}

$catTable = DB_PREFIX . 'category';
if (!tableExists($m, $catTable)) {
  throw new Exception('category table missing');
}

$checkOld = $m->query("SELECT category_id FROM `" . $catTable . "` WHERE category_id=$oldId");
$checkNew = $m->query("SELECT category_id FROM `" . $catTable . "` WHERE category_id=$newId");
if (!$checkOld || $checkOld->num_rows === 0) {
  throw new Exception("Old category id $oldId not found");
}
if ($checkNew && $checkNew->num_rows > 0) {
  throw new Exception("Target category id $newId already exists");
}

$m->begin_transaction();

try {
  // Primary category table
  $m->query("UPDATE `" . $catTable . "` SET category_id=$newId WHERE category_id=$oldId");
  if (colExists($m, $catTable, 'parent_id')) {
    $m->query("UPDATE `" . $catTable . "` SET parent_id=$newId WHERE parent_id=$oldId");
  }

  $tablesWithCategoryId = [
    DB_PREFIX . 'category_description',
    DB_PREFIX . 'category_filter',
    DB_PREFIX . 'category_to_layout',
    DB_PREFIX . 'category_to_store',
    DB_PREFIX . 'product_to_category'
  ];

  foreach ($tablesWithCategoryId as $t) {
    if (tableExists($m, $t) && colExists($m, $t, 'category_id')) {
      $m->query("UPDATE `" . $t . "` SET category_id=$newId WHERE category_id=$oldId");
    }
  }

  $pathTable = DB_PREFIX . 'category_path';
  if (tableExists($m, $pathTable)) {
    if (colExists($m, $pathTable, 'category_id')) {
      $m->query("UPDATE `" . $pathTable . "` SET category_id=$newId WHERE category_id=$oldId");
    }
    if (colExists($m, $pathTable, 'path_id')) {
      $m->query("UPDATE `" . $pathTable . "` SET path_id=$newId WHERE path_id=$oldId");
    }
  }

  $urlAlias = DB_PREFIX . 'url_alias';
  if (tableExists($m, $urlAlias) && colExists($m, $urlAlias, 'query')) {
    $m->query("UPDATE `" . $urlAlias . "` SET `query`='category_id=$newId' WHERE `query`='category_id=$oldId'");
  }

  $seoUrl = DB_PREFIX . 'seo_url';
  if (tableExists($m, $seoUrl) && colExists($m, $seoUrl, 'query')) {
    $m->query("UPDATE `" . $seoUrl . "` SET `query`='category_id=$newId' WHERE `query`='category_id=$oldId'");
  }

  $m->commit();

  // Verify
  $q = $m->query("SELECT c.category_id, cd.name FROM `" . DB_PREFIX . "category` c LEFT JOIN `" . DB_PREFIX . "category_description` cd ON cd.category_id=c.category_id AND cd.language_id=1 WHERE c.category_id=$newId");
  $row = $q ? $q->fetch_assoc() : null;
  echo "updated_to_id=" . ($row['category_id'] ?? 'none') . " name=" . ($row['name'] ?? 'NULL') . PHP_EOL;

} catch (Throwable $e) {
  $m->rollback();
  throw $e;
}