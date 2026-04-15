<?php
require 'c:/xampp/htdocs/opencart1/config.php';

$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');

$prefixLike = "%Imported from products_2026.csv%";
$r = $m->query("SELECT product_id, language_id, description FROM " . DB_PREFIX . "product_description WHERE description LIKE '" . $m->real_escape_string($prefixLike) . "'");

$updated = 0;
while ($row = $r->fetch_assoc()) {
    $desc = (string)$row['description'];

    // Remove the generated import header and handle line at the start of description.
    $desc = preg_replace('/^\s*Imported from products_2026\.csv\s*Handle:\s*[^\r\n]*\s*/i', '', $desc);

    // Fallback for single-line variant if newline was collapsed in output.
    $desc = str_replace('Imported from products_2026.csv Handle:', '', $desc);

    $desc = trim($desc);

    $product_id = (int)$row['product_id'];
    $language_id = (int)$row['language_id'];
    $descEsc = $m->real_escape_string($desc);

    $m->query("UPDATE " . DB_PREFIX . "product_description SET description='" . $descEsc . "' WHERE product_id=$product_id AND language_id=$language_id");
    $updated++;
}

echo "updated_rows=" . $updated . PHP_EOL;