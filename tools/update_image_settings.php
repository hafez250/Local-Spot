<?php
require 'c:/xampp/htdocs/opencart1/config.php';
$m = new mysqli(DB_HOSTNAME, DB_USERNAME, DB_PASSWORD, DB_DATABASE, (int)DB_PORT);
$m->set_charset('utf8mb4');
$updates = [
  'config_image_product_width' => '500',
  'config_image_product_height' => '500',
  'config_image_thumb_width' => '800',
  'config_image_thumb_height' => '800',
  'config_image_additional_width' => '120',
  'config_image_additional_height' => '120'
];
foreach ($updates as $k => $v) {
  $kEsc = $m->real_escape_string($k);
  $vEsc = $m->real_escape_string($v);
  $m->query("UPDATE " . DB_PREFIX . "setting SET `value`='$vEsc' WHERE `key`='$kEsc'");
}
echo "image settings updated\n";