<?php
namespace Opencart\Catalog\Controller\Common;
/**
 * Class Home
 *
 * Can be called from $this->load->controller('common/home');
 *
 * @package Opencart\Catalog\Controller\Common
 */
class Home extends \Opencart\System\Engine\Controller {
	/**
	 * Index
	 *
	 * @return void
	 */
	public function index(): void {
		$description = $this->config->get('config_description');
		$language_id = $this->config->get('config_language_id');

		if (isset($description[$language_id])) {
			$this->document->setTitle($description[$language_id]['meta_title']);
			$this->document->setDescription($description[$language_id]['meta_description']);
			$this->document->setKeywords($description[$language_id]['meta_keyword']);
		}

		$data['column_left'] = $this->load->controller('common/column_left');
		$data['column_right'] = $this->load->controller('common/column_right');
		$data['content_top'] = $this->load->controller('common/content_top');
		$data['content_bottom'] = $this->load->controller('common/content_bottom');
		$data['footer'] = $this->load->controller('common/footer');
		$data['header'] = $this->load->controller('common/header');
        $data['search'] = $this->load->controller('common/search');
		$data['cart'] = $this->load->controller('common/cart');
		$data['menu'] = $this->load->controller('common/menu');
        $data['route'] = $this->request->get['route'] ?? 'common/home';
		$language = 'language=' . $this->config->get('config_language');
		$data['category_clothing'] = $this->url->link('product/category', $language . '&path=492');
		$data['category_electronics'] = $this->url->link('product/category', $language . '&path=59');
		$data['category_accessories'] = $this->url->link('product/category', $language . '&path=60');
		$data['category_starnine'] = $this->url->link('product/category', $language . '&path=61');
		$data['category_seemly'] = $this->url->link('product/category', $language . '&path=493');
		$data['category_antikka'] = $this->url->link('product/category', $language . '&path=504');
		$data['category_svg'] = $this->url->link('product/category', $language . '&path=62');
		$data['category_basiclook'] = $this->url->link('product/category', $language . '&path=516');
		$data['category_kntd'] = $this->url->link('product/category', $language . '&path=527');
		$data['category_corebasics'] = $this->url->link('product/category', $language . '&path=544');
		$data['category_footwear'] = $this->url->link('product/category', $language . '&path=63');
		$data['category_men'] = $this->url->link('product/men', $language);
		$data['category_woman'] = $this->url->link('product/women', $language);
		$data['category_winter'] = $this->url->link('product/winter', $language);
		$data['category_summer'] = $this->url->link('product/summer', $language);
		$data['category_brands'] = $this->url->link('product/manufacturer', $language);
        $data['path'] = isset($this->request->get['path']) ? $this->request->get['path'] : '';

		$local_brands = [
			[
				'category_id' => 61,
				'name'        => 'Starnine'
			],
			[
				'category_id' => 62,
				'name'        => 'Svg'
			],
			[
				'category_id' => 493,
				'name'        => 'Seemly'
			],
			[
				'category_id' => 504,
				'name'        => 'Antikka'
			]
		];

		$winter_keywords = ['hoodie', 'hoodies', 'zipper', 'jacket', 'crewneck', 'sweatshirt', 'sweater', 'coat', 'winter'];
		$summer_keywords = ['top', 'tops', 'tee', 'tees', 't-shirt', 'tshirt', 'short', 'shorts', 'tank', 'summer'];
		$winter_exclude_keywords = ['tee', 'tees', 't-shirt', 'tshirt', 'top', 'tops', 'tank', 'short', 'shorts', 'shirt'];
		$summer_exclude_keywords = ['hoodie', 'hoodies', 'zipper', 'jacket', 'crewneck', 'sweatshirt', 'sweater', 'coat', 'winter'];

		$winter_forced_by_brand = [
			'Starnine' => ['stitched crewneck']
		];
		$summer_forced_by_brand = [
			'Svg'    => ['cloudy shirt'],
			'Seemly' => ['pink spaghetti top']
		];

		$data['winter_products'] = $this->getSeasonProducts($local_brands, $winter_keywords, $winter_forced_by_brand, $winter_exclude_keywords);
		$data['summer_products'] = $this->getSeasonProducts($local_brands, $summer_keywords, $summer_forced_by_brand, $summer_exclude_keywords);

		$data['winter_explore_all'] = $this->url->link(
			'product/winter',
			$language
		);
		$data['summer_explore_all'] = $this->url->link(
			'product/summer',
			$language
		);

		$this->response->setOutput($this->load->view('common/home', $data));
	}

	/**
	 * Build season product cards by selecting one product from each local brand.
	 *
	 * @param array<int, array<string, mixed>> $brands
	 * @param array<int, string> $keywords
	 *
	 * @return array<int, array<string, mixed>>
	 */
	private function getSeasonProducts(array $brands, array $keywords, array $forcedByBrand = [], array $excludeKeywords = []): array {
		$this->load->model('catalog/product');
		$this->load->model('tool/image');

		$products = [];

		foreach ($brands as $brand) {
			$results = $this->model_catalog_product->getProducts([
				'filter_category_id'  => (int)$brand['category_id'],
				'filter_sub_category' => true,
				'sort'                => 'p.date_added',
				'order'               => 'DESC',
				'start'               => 0,
				'limit'               => 50
			]);

			if (!$results) {
				continue;
			}

			$forced_terms = [];

			if (!empty($forcedByBrand[$brand['name']])) {
				$forced_terms = $forcedByBrand[$brand['name']];
			}

			$selected_product = $this->pickSeasonProduct($results, $keywords, $forced_terms, $excludeKeywords);

			if (!$selected_product) {
				continue;
			}

			if (!empty($selected_product['image']) && is_file(DIR_IMAGE . html_entity_decode((string)$selected_product['image'], ENT_QUOTES, 'UTF-8'))) {
				$image = (string)$selected_product['image'];
			} else {
				$image = 'placeholder.png';
			}

			if ($this->customer->isLogged() || !$this->config->get('config_customer_price')) {
				$price = $this->currency->format(
					$this->tax->calculate((float)$selected_product['price'], (int)$selected_product['tax_class_id'], $this->config->get('config_tax')),
					$this->session->data['currency']
				);
			} else {
				$price = false;
			}

			if ((float)$selected_product['special']) {
				$special = $this->currency->format(
					$this->tax->calculate((float)$selected_product['special'], (int)$selected_product['tax_class_id'], $this->config->get('config_tax')),
					$this->session->data['currency']
				);
			} else {
				$special = false;
			}

			$products[] = [
				'product_id' => (int)$selected_product['product_id'],
				'name'       => (string)$selected_product['name'],
				'thumb'      => $this->model_tool_image->resize($image, 400, 400),
				'price'      => $price,
				'special'    => $special,
				'brand_name' => (string)$brand['name'],
				'brand_href' => $this->url->link(
					'product/category',
					'language=' . $this->config->get('config_language') . '&path=492_' . (int)$brand['category_id']
				),
				'href'       => $this->url->link(
					'product/product',
					'language=' . $this->config->get('config_language') . '&product_id=' . (int)$selected_product['product_id']
				)
			];
		}

		return array_slice($products, 0, 4);
	}

	/**
	 * Pick the first product that matches at least one season keyword.
	 *
	 * @param array<int, array<string, mixed>> $products
	 * @param array<int, string> $keywords
	 *
	 * @return array<string, mixed>
	 */
	private function pickSeasonProduct(array $products, array $keywords, array $forcedTerms = [], array $excludeKeywords = []): array {
		if ($forcedTerms) {
			foreach ($products as $product) {
				$name = oc_strtolower((string)$product['name']);
				$description = oc_strtolower(strip_tags(html_entity_decode((string)$product['description'], ENT_QUOTES, 'UTF-8')));
				$haystack = $name . ' ' . $description;

				if ($this->containsAny($haystack, $excludeKeywords)) {
					continue;
				}

				foreach ($forcedTerms as $term) {
					if (strpos($haystack, oc_strtolower($term)) !== false) {
						return $product;
					}
				}
			}
		}

		foreach ($products as $product) {
			$name = oc_strtolower((string)$product['name']);
			$description = oc_strtolower(strip_tags(html_entity_decode((string)$product['description'], ENT_QUOTES, 'UTF-8')));
			$haystack = $name . ' ' . $description;

			if ($this->containsAny($haystack, $excludeKeywords)) {
				continue;
			}

			foreach ($keywords as $keyword) {
				if (strpos($haystack, oc_strtolower($keyword)) !== false) {
					return $product;
				}
			}
		}

		return [];
	}

	/**
	 * @param string $haystack
	 * @param array<int, string> $terms
	 *
	 * @return bool
	 */
	private function containsAny(string $haystack, array $terms): bool {
		foreach ($terms as $term) {
			if (strpos($haystack, oc_strtolower($term)) !== false) {
				return true;
			}
		}

		return false;
	}
}
