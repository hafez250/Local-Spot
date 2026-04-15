<?php
namespace Opencart\Catalog\Controller\Common;
/**
 * Class Search
 *
 * Can be called from $this->load->controller('common/search');
 *
 * @package Opencart\Catalog\Controller\Common
 */
class Search extends \Opencart\System\Engine\Controller {
	/**
	 * Index
	 *
	 * @return string
	 */
	public function index(): string {
		$this->load->language('common/search');

		$data['text_search'] = $this->language->get('text_search');

		$data['action'] = $this->url->link('common/search.redirect', 'language=' . $this->config->get('config_language'));
		$data['autocomplete'] = $this->url->link('common/search.autocomplete', 'language=' . $this->config->get('config_language'));

		if (isset($this->request->get['search'])) {
			$data['search'] = $this->request->get['search'];
		} else {
			$data['search'] = '';
		}

		return $this->load->view('common/search', $data);
	}

	/**
	 * Product autocomplete
	 *
	 * @return void
	 */
	public function autocomplete(): void {
		$json = [];

		$search = isset($this->request->get['term']) ? trim((string)$this->request->get['term']) : '';

		if (oc_strlen($search) < 1) {
			$this->response->addHeader('Content-Type: application/json');
			$this->response->setOutput(json_encode($json));
			return;
		}

		$this->load->model('catalog/product');
		$this->load->model('tool/image');

		$results = $this->model_catalog_product->getProducts([
			'filter_search' => $search,
			'sort'          => 'p.date_added',
			'order'         => 'DESC',
			'start'         => 0,
			'limit'         => 8
		]);

		foreach ($results as $result) {
			if (!empty($result['image']) && is_file(DIR_IMAGE . html_entity_decode((string)$result['image'], ENT_QUOTES, 'UTF-8'))) {
				$image = (string)$result['image'];
			} else {
				$image = 'placeholder.png';
			}

			if ($this->customer->isLogged() || !$this->config->get('config_customer_price')) {
				$price = $this->currency->format(
					$this->tax->calculate((float)$result['price'], (int)$result['tax_class_id'], $this->config->get('config_tax')),
					$this->session->data['currency']
				);
			} else {
				$price = '';
			}

			$json[] = [
				'product_id' => (int)$result['product_id'],
				'name'       => (string)$result['name'],
				'thumb'      => $this->model_tool_image->resize($image, 64, 64),
				'price'      => $price,
				'href'       => $this->url->link(
					'product/product',
					'language=' . $this->config->get('config_language') . '&product_id=' . (int)$result['product_id']
				)
			];
		}

		$this->response->addHeader('Content-Type: application/json');
		$this->response->setOutput(json_encode($json));
	}

	/**
	 * Redirect
	 */
	public function redirect(): void {
		if (isset($this->request->post['search'])) {
			$search = urlencode(html_entity_decode($this->request->post['search'], ENT_QUOTES, 'UTF-8'));
		} else {
			$search = '';
		}

		$this->response->redirect($this->url->link('product/search', 'language=' . $this->config->get('config_language') . '&search=' . $search, true));
	}
}
