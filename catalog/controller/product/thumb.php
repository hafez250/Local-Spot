<?php
namespace Opencart\Catalog\Controller\Product;
/**
 * Class Thumb
 *
 * Can be loaded using $this->load->controller('product/thumb', $product_data);
 *
 * @example
 *
 * $product_data = [
 *     'description' => '',
 *     'thumb'       => '',
 *     'price'       => 1.00,
 *     'special'     => 0.00,
 *     'tax'         => 0.00,
 *     'minimum'     => 1,
 *     'href'        => ''
 * ];
 *
 * @package Opencart\Catalog\Controller\Product
 */
class Thumb extends \Opencart\System\Engine\Controller {
	/**
	 * Index
	 *
	 * @param array<string, mixed> $data array of data
	 *
	 * @return string
	 */
	public function index(array $data): string {
		$this->load->language('product/thumb');
		$this->load->model('tool/image');
		$this->load->model('catalog/manufacturer');

		$data['cart'] = $this->url->link('common/cart.info', 'language=' . $this->config->get('config_language'));

		$data['cart_add'] = $this->url->link('checkout/cart.add', 'language=' . $this->config->get('config_language'));
		$data['wishlist_add'] = $this->url->link('account/wishlist.add', 'language=' . $this->config->get('config_language'));
		$data['wishlist_remove'] = $this->url->link('account/wishlist.remove', 'language=' . $this->config->get('config_language'));
		$data['compare_add'] = $this->url->link('product/compare.add', 'language=' . $this->config->get('config_language'));

		$data['review_status'] = (int)$this->config->get('config_review_status');

		// Ensure every product card has a valid image URL.
		if (empty($data['thumb'])) {
			$data['thumb'] = $this->model_tool_image->resize(
				'placeholder.png',
				$this->config->get('config_image_product_width'),
				$this->config->get('config_image_product_height')
			);
		}

		$product_id = isset($data['product_id']) ? (int)$data['product_id'] : 0;
		$wishlist_product_ids = [];

		if ($this->customer->isLogged()) {
			$this->load->model('account/wishlist');

			foreach ($this->model_account_wishlist->getWishlist($this->customer->getId()) as $wishlist_product) {
				$wishlist_product_ids[] = (int)$wishlist_product['product_id'];
			}
		} elseif (!empty($this->session->data['wishlist']) && is_array($this->session->data['wishlist'])) {
			$wishlist_product_ids = array_map('intval', $this->session->data['wishlist']);
		}

		$data['in_wishlist'] = in_array($product_id, $wishlist_product_ids, true);

		$data['brand_name'] = isset($data['brand_name']) ? trim((string)$data['brand_name']) : '';
		$data['brand_href'] = isset($data['brand_href']) ? (string)$data['brand_href'] : '';

		if ($data['brand_name'] === '' && !empty($data['manufacturer'])) {
			$data['brand_name'] = (string)$data['manufacturer'];
		} elseif ($data['brand_name'] === '' && !empty($data['manufacturer_id'])) {
			$manufacturer_info = $this->model_catalog_manufacturer->getManufacturer((int)$data['manufacturer_id']);

			if ($manufacturer_info) {
				$data['brand_name'] = (string)$manufacturer_info['name'];
			}
		}

		if ($data['brand_href'] === '' && !empty($data['manufacturer_id'])) {
			$data['brand_href'] = $this->url->link(
				'product/manufacturer.info',
				'language=' . $this->config->get('config_language') . '&manufacturer_id=' . (int)$data['manufacturer_id']
			);
		}

		return $this->load->view('product/thumb', $data);
	}
}
