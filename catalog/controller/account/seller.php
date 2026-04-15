<?php
namespace Opencart\Catalog\Controller\Account;
/**
 * Seller Dashboard
 *
 * @package Opencart\Catalog\Controller\Account
 */
class Seller extends \Opencart\System\Engine\Controller {
	public function index(): void {
		if (!$this->load->controller('account/login.validate')) {
			$this->session->data['redirect'] = $this->url->link('account/seller', 'language=' . $this->config->get('config_language'));
			$this->response->redirect($this->url->link('account/login', 'language=' . $this->config->get('config_language'), true));
		}

		if (!$this->isSeller()) {
			$this->response->redirect($this->url->link('account/account', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}

		$this->load->language('account/seller');
		$this->document->setTitle($this->language->get('heading_title'));

		$this->ensureSellerTable();

		$customer_id = (int)$this->customer->getId();
		$seller = $this->getSellerProfile($customer_id);
		$is_new_brand = empty($seller['brand_name']);
		$needs_onboarding = empty($seller['brand_name']);

		$data['breadcrumbs'] = [];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_home'),
			'href' => $this->url->link('common/home', 'language=' . $this->config->get('config_language'))
		];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_account'),
			'href' => $this->url->link('account/account', 'language=' . $this->config->get('config_language'))
		];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_seller'),
			'href' => $this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'])
		];

		$data['brand_name'] = $seller['brand_name'] ?? '';
		$data['brand_image'] = $seller['brand_image'] ?? '';
		$data['brand_gender'] = $seller['brand_gender'] ?? '';
		$data['gender_options'] = [
			'women' => $this->language->get('text_gender_women'),
			'men'   => $this->language->get('text_gender_men'),
			'both'  => $this->language->get('text_gender_both')
		];
		$data['brand_save'] = $this->url->link('account/seller.saveBrand', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token']);
		$data['product_save'] = $this->url->link('account/seller.saveProduct', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token']);

		$data['success'] = $this->session->data['success'] ?? '';
		$data['error'] = $this->session->data['error'] ?? '';
		unset($this->session->data['success'], $this->session->data['error']);

		$data['categories'] = $this->getCategories();

		$data['column_left'] = $this->load->controller('common/column_left');
		$data['column_right'] = $this->load->controller('common/column_right');
		$data['content_top'] = $this->load->controller('common/content_top');
		$data['content_bottom'] = $this->load->controller('common/content_bottom');
		$data['footer'] = $this->load->controller('common/footer');
		$data['header'] = $this->load->controller('common/header');

		if ($needs_onboarding) {
			$this->response->setOutput($this->load->view('account/seller_onboard', $data));
			return;
		}

		$this->response->setOutput($this->load->view('account/seller', $data));
	}

	public function saveBrand(): void {
		$this->load->language('account/seller');

		if (!$this->load->controller('account/login.validate') || !$this->isSeller()) {
			$this->response->redirect($this->url->link('account/login', 'language=' . $this->config->get('config_language'), true));
		}

		$this->ensureSellerTable();

		$customer_id = (int)$this->customer->getId();
		$brand_name = trim((string)($this->request->post['brand_name'] ?? ''));
		$brand_gender = (string)($this->request->post['brand_gender'] ?? '');
		$allowed_gender = ['women', 'men', 'both'];

		if ($brand_name === '') {
			$this->session->data['error'] = $this->language->get('error_brand_name');
			$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}
		if (!in_array($brand_gender, $allowed_gender, true)) {
			$this->session->data['error'] = $this->language->get('error_brand_gender');
			$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}

		$seller = $this->getSellerProfile($customer_id);
		$manufacturer_id = isset($seller['manufacturer_id']) ? (int)$seller['manufacturer_id'] : 0;

		$image_path = $this->saveLogoUpload($customer_id);
		if ($image_path === '') {
			$image_path = $seller['brand_image'] ?? '';
		}

		if ($manufacturer_id) {
			$this->db->query("UPDATE `" . DB_PREFIX . "manufacturer` SET name = '" . $this->db->escape($brand_name) . "', image = '" . $this->db->escape($image_path) . "' WHERE manufacturer_id = '" . (int)$manufacturer_id . "'");
		} else {
			$this->db->query("INSERT INTO `" . DB_PREFIX . "manufacturer` SET name = '" . $this->db->escape($brand_name) . "', image = '" . $this->db->escape($image_path) . "', sort_order = '0'");
			$manufacturer_id = (int)$this->db->getLastId();
			$this->db->query("INSERT IGNORE INTO `" . DB_PREFIX . "manufacturer_to_store` SET manufacturer_id = '" . $manufacturer_id . "', store_id = '0'");
		}

		if ($seller) {
			$this->db->query("UPDATE `" . DB_PREFIX . "seller` SET manufacturer_id = '" . (int)$manufacturer_id . "', brand_name = '" . $this->db->escape($brand_name) . "', brand_image = '" . $this->db->escape($image_path) . "', brand_gender = '" . $this->db->escape($brand_gender) . "' WHERE customer_id = '" . $customer_id . "'");
		} else {
			$this->db->query("INSERT INTO `" . DB_PREFIX . "seller` SET customer_id = '" . $customer_id . "', manufacturer_id = '" . (int)$manufacturer_id . "', brand_name = '" . $this->db->escape($brand_name) . "', brand_image = '" . $this->db->escape($image_path) . "', brand_gender = '" . $this->db->escape($brand_gender) . "'");
		}

		$this->session->data['success'] = $this->language->get('text_brand_saved');
		if ($is_new_brand) {
			$this->session->data['seller_welcome'] = 1;
			$this->response->redirect($this->url->link('account/seller.welcome', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		} else {
			$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}
	}

	public function saveProduct(): void {
		$this->load->language('account/seller');

		if (!$this->load->controller('account/login.validate') || !$this->isSeller()) {
			$this->response->redirect($this->url->link('account/login', 'language=' . $this->config->get('config_language'), true));
		}

		$this->ensureSellerTable();

		$customer_id = (int)$this->customer->getId();
		$seller = $this->getSellerProfile($customer_id);
		$manufacturer_id = isset($seller['manufacturer_id']) ? (int)$seller['manufacturer_id'] : 0;

		if (!$manufacturer_id) {
			$this->session->data['error'] = $this->language->get('error_brand_required');
			$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}

		$name = trim((string)($this->request->post['product_name'] ?? ''));
		$price = (float)($this->request->post['product_price'] ?? 0);
		$description = trim((string)($this->request->post['product_description'] ?? ''));
		$category_id = (int)($this->request->post['category_id'] ?? 0);

		if ($name === '' || $price <= 0 || $category_id <= 0) {
			$this->session->data['error'] = $this->language->get('error_product_fields');
			$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}

		$image_path = $this->saveProductImage($customer_id, $name);

		$stock_status_id = $this->getInStockStatusId();
		$sku = $this->makeSku($customer_id, $name);

		$this->db->query("INSERT INTO `" . DB_PREFIX . "product` SET model = '" . $this->db->escape($name) . "', sku = '" . $this->db->escape($sku) . "', upc = '', ean = '', jan = '', isbn = '', mpn = '', location = '', quantity = '100', stock_status_id = '" . (int)$stock_status_id . "', image = '" . $this->db->escape($image_path) . "', manufacturer_id = '" . (int)$manufacturer_id . "', shipping = '1', price = '" . (float)$price . "', points = '0', tax_class_id = '0', date_available = NOW(), weight = '0', weight_class_id = '0', length = '0', width = '0', height = '0', length_class_id = '0', subtract = '1', minimum = '1', sort_order = '0', status = '1', date_added = NOW(), date_modified = NOW()");
		$product_id = (int)$this->db->getLastId();

		$this->db->query("INSERT INTO `" . DB_PREFIX . "product_description` SET product_id = '" . $product_id . "', language_id = '" . (int)$this->config->get('config_language_id') . "', name = '" . $this->db->escape($name) . "', description = '" . $this->db->escape($description) . "', tag = '', meta_title = '" . $this->db->escape($name) . "', meta_description = '', meta_keyword = ''");
		$this->db->query("INSERT IGNORE INTO `" . DB_PREFIX . "product_to_store` SET product_id = '" . $product_id . "', store_id = '0'");
		$this->db->query("INSERT IGNORE INTO `" . DB_PREFIX . "product_to_category` SET product_id = '" . $product_id . "', category_id = '" . (int)$category_id . "'");

		$this->session->data['success'] = $this->language->get('text_product_saved');
		$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
	}

	public function welcome(): void {
		if (!$this->load->controller('account/login.validate')) {
			$this->session->data['redirect'] = $this->url->link('account/seller.welcome', 'language=' . $this->config->get('config_language'));
			$this->response->redirect($this->url->link('account/login', 'language=' . $this->config->get('config_language'), true));
		}

		if (!$this->isSeller()) {
			$this->response->redirect($this->url->link('account/account', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}

		if (empty($this->session->data['seller_welcome'])) {
			$this->response->redirect($this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true));
		}

		unset($this->session->data['seller_welcome']);

		$this->load->language('account/seller');
		$this->document->setTitle($this->language->get('heading_welcome'));
		$this->ensureSellerTable();

		$customer_id = (int)$this->customer->getId();
		$seller = $this->getSellerProfile($customer_id);

		$this->load->model('tool/image');

		$data['brand_name'] = $seller['brand_name'] ?? '';
		$data['brand_logo'] = '';

		if (!empty($seller['brand_image']) && is_file(DIR_IMAGE . $seller['brand_image'])) {
			$data['brand_logo'] = $this->model_tool_image->resize($seller['brand_image'], 240, 240);
		}

		$data['continue'] = $this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'], true);

		$data['breadcrumbs'] = [];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_home'),
			'href' => $this->url->link('common/home', 'language=' . $this->config->get('config_language'))
		];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_account'),
			'href' => $this->url->link('account/account', 'language=' . $this->config->get('config_language'))
		];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_seller'),
			'href' => $this->url->link('account/seller', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token'])
		];

		$data['column_left'] = $this->load->controller('common/column_left');
		$data['column_right'] = $this->load->controller('common/column_right');
		$data['content_top'] = $this->load->controller('common/content_top');
		$data['content_bottom'] = $this->load->controller('common/content_bottom');
		$data['footer'] = $this->load->controller('common/footer');
		$data['header'] = $this->load->controller('common/header');

		$this->response->setOutput($this->load->view('account/seller_welcome', $data));
	}

	private function isSeller(): bool {
		return $this->customer->isLogged() && $this->customer->getGroupId() === $this->getSellerGroupId();
	}

	private function getSellerGroupId(): int {
		$seller_name = 'Seller';
		$language_id = (int)$this->config->get('config_language_id');

		$query = $this->db->query("SELECT customer_group_id FROM `" . DB_PREFIX . "customer_group_description` WHERE language_id = '" . $language_id . "' AND name = '" . $this->db->escape($seller_name) . "' LIMIT 1");

		if ($query->num_rows) {
			return (int)$query->row['customer_group_id'];
		}

		$this->db->query("INSERT INTO `" . DB_PREFIX . "customer_group` SET approval = '0', sort_order = '0'");
		$customer_group_id = (int)$this->db->getLastId();

		$this->db->query("INSERT INTO `" . DB_PREFIX . "customer_group_description` SET customer_group_id = '" . $customer_group_id . "', language_id = '" . $language_id . "', name = '" . $this->db->escape($seller_name) . "', description = ''");

		return $customer_group_id;
	}

	private function ensureSellerTable(): void {
		$this->db->query("CREATE TABLE IF NOT EXISTS `" . DB_PREFIX . "seller` (
			`customer_id` int(11) NOT NULL,
			`manufacturer_id` int(11) DEFAULT NULL,
			`brand_name` varchar(64) DEFAULT NULL,
			`brand_image` varchar(255) DEFAULT NULL,
			`brand_gender` varchar(16) DEFAULT NULL,
			PRIMARY KEY (`customer_id`)
		) ENGINE=MyISAM DEFAULT CHARSET=utf8mb4");

		$check = $this->db->query("SHOW COLUMNS FROM `" . DB_PREFIX . "seller` LIKE 'brand_gender'");
		if (!$check->num_rows) {
			$this->db->query("ALTER TABLE `" . DB_PREFIX . "seller` ADD COLUMN `brand_gender` varchar(16) DEFAULT NULL");
		}
	}

	private function getSellerProfile(int $customer_id): array {
		$query = $this->db->query("SELECT * FROM `" . DB_PREFIX . "seller` WHERE customer_id = '" . (int)$customer_id . "' LIMIT 1");
		return $query->num_rows ? $query->row : [];
	}

	private function saveLogoUpload(int $customer_id): string {
		if (!isset($this->request->files['brand_logo']) || !is_uploaded_file($this->request->files['brand_logo']['tmp_name'])) {
			return '';
		}

		$ext = strtolower(pathinfo($this->request->files['brand_logo']['name'], PATHINFO_EXTENSION));
		if (!in_array($ext, ['jpg', 'jpeg', 'png', 'webp', 'avif'], true)) {
			return '';
		}

		$dir = DIR_IMAGE . 'catalog/brands/';
		if (!is_dir($dir)) {
			mkdir($dir, 0777, true);
		}

		$filename = 'seller-' . $customer_id . '.' . $ext;
		$dest = $dir . $filename;

		if (move_uploaded_file($this->request->files['brand_logo']['tmp_name'], $dest)) {
			return 'catalog/brands/' . $filename;
		}

		return '';
	}

	private function saveProductImage(int $customer_id, string $name): string {
		if (!isset($this->request->files['product_image']) || !is_uploaded_file($this->request->files['product_image']['tmp_name'])) {
			return '';
		}

		$ext = strtolower(pathinfo($this->request->files['product_image']['name'], PATHINFO_EXTENSION));
		if (!in_array($ext, ['jpg', 'jpeg', 'png', 'webp', 'avif'], true)) {
			return '';
		}

		$dir = DIR_IMAGE . 'catalog/products/';
		if (!is_dir($dir)) {
			mkdir($dir, 0777, true);
		}

		$safe = preg_replace('/[^a-zA-Z0-9_-]/', '-', oc_strtolower($name));
		$filename = 'seller-' . $customer_id . '-' . $safe . '-' . time() . '.' . $ext;
		$dest = $dir . $filename;

		if (move_uploaded_file($this->request->files['product_image']['tmp_name'], $dest)) {
			return 'catalog/products/' . $filename;
		}

		return '';
	}

	private function makeSku(int $customer_id, string $name): string {
		$base = preg_replace('/[^a-zA-Z0-9_-]/', '-', oc_strtolower($name));
		$base = trim($base, '-');
		if ($base === '') {
			$base = 'seller-item';
		}
		return substr($base . '-' . $customer_id . '-' . time(), 0, 64);
	}

	private function getInStockStatusId(): int {
		$status_id = 7;
		$query = $this->db->query("SELECT stock_status_id FROM `" . DB_PREFIX . "stock_status` WHERE name = 'In Stock' LIMIT 1");
		if ($query->num_rows) {
			$status_id = (int)$query->row['stock_status_id'];
		}
		return $status_id;
	}

	private function getCategories(): array {
		$language_id = (int)$this->config->get('config_language_id');
		$query = $this->db->query("SELECT c.category_id, cd.name FROM `" . DB_PREFIX . "category` c INNER JOIN `" . DB_PREFIX . "category_description` cd ON cd.category_id = c.category_id WHERE cd.language_id = '" . $language_id . "' ORDER BY cd.name ASC");
		return $query->rows;
	}
}
