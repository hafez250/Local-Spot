<?php
namespace Opencart\Catalog\Controller\Api;
/**
 * Class Recommendation
 *
 * Recommendation API
 *
 * @package Opencart\Catalog\Controller\Api
 */
class Recommendation extends \Opencart\System\Engine\Controller {
	/**
	 * Index
	 *
	 * @return void
	 */
	public function index(): void {
		$this->load->language('api/recommendation');

		if (isset($this->request->get['call'])) {
			$call = (string)$this->request->get['call'];
		} else {
			$call = 'get';
		}

		switch ($call) {
			case 'get':
				$output = $this->getRecommendations();
				break;
			default:
				$output = ['error' => $this->language->get('error_call')];
				break;
		}

		$this->response->addHeader('Content-Type: application/json');
		$this->response->setOutput(json_encode($output));
	}

	/**
	 * Get recommendations
	 *
	 * @return array
	 */
	protected function getRecommendations(): array {
		$customer_id = (int)($this->request->post['customer_id'] ?? $this->request->get['customer_id'] ?? 0);
		$limit = (int)($this->request->post['limit'] ?? $this->request->get['limit'] ?? 5);

		if ($limit < 1) {
			$limit = 5;
		} elseif ($limit > 20) {
			$limit = 20;
		}

		if ($customer_id <= 0) {
			return ['error' => $this->language->get('error_customer_id')];
		}

		$this->load->model('module/suggestion');

		$products = $this->model_module_suggestion->getSuggestedProducts($customer_id, false, $limit);

		if ($products && count($products) > $limit) {
			$products = array_slice($products, 0, $limit);
		}

		return [
			'customer_id' => $customer_id,
			'count' => count($products),
			'products' => $products
		];
	}
}
