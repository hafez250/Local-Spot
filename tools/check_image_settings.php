<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$keys = [
 'config_image_product_width','config_image_product_height',
 'config_image_thumb_width','config_image_thumb_height',
 'config_image_additional_width','config_image_additional_height',
 'theme_default_image_product_width','theme_default_image_product_height',
 'theme_default_image_thumb_width','theme_default_image_thumb_height'
];
foreach ($keys as $k) {
  $kEsc = $m->real_escape_string($k);
  $r = $m->query("SELECT `value` FROM " . DB_PREFIX . "setting WHERE `key`='$kEsc' ORDER BY store_id LIMIT 1");
  $v = ($r && $row=$r->fetch_assoc()) ? $row['value'] : 'N/A';
  echo $k . '=' . $v . PHP_EOL;
}