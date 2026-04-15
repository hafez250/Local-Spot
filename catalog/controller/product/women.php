<?php
namespace Opencart\Catalog\Controller\Product;

/**
 * Women's Collection page with keyword and brand filters.
 */
class Women extends \Opencart\System\Engine\Controller {
	/**
	 * Index
	 *
	 * @return void
	 */
	public function index(): void {
		$this->load->language('product/category');
		$this->load->model('catalog/product');
		$this->load->model('catalog/manufacturer');
		$this->load->model('tool/image');

		$sort = $this->request->get['sort'] ?? 'p.date_added';
		$order = $this->request->get['order'] ?? 'DESC';
		$page = isset($this->request->get['page']) ? (int)$this->request->get['page'] : 1;
		$limit = (isset($this->request->get['limit']) && (int)$this->request->get['limit']) ? (int)$this->request->get['limit'] : 12;

		$filter_options = [
			'hoodie'   => 'Hoodies',
			'zipper'   => 'Zipper',
			'jacket'   => 'Jackets',
			'crewneck' => 'Crewneck',
			'tee'      => 'Tees',
			'top'      => 'Tops',
			'shirt'    => 'Shirts',
			'pants'    => 'Pants',
			'shorts'   => 'Shorts'
		];
		$brand_filter_options = $this->getBrandFilterOptions();

		$selected_tags = [];
		$selected_brand_ids = [];

		if (!empty($this->request->get['women_tags'])) {
			foreach (explode(',', (string)$this->request->get['women_tags']) as $tag) {
				$key = trim(oc_strtolower($tag));

				if (isset($filter_options[$key])) {
					$selected_tags[] = $key;
				}
			}

			$selected_tags = array_values(array_unique($selected_tags));
		}

		if (!empty($this->request->get['women_brands'])) {
			foreach (explode(',', (string)$this->request->get['women_brands']) as $brand_id) {
				$id = (int)$brand_id;

				if (isset($brand_filter_options[$id])) {
					$selected_brand_ids[] = $id;
				}
			}

			$selected_brand_ids = array_values(array_unique($selected_brand_ids));
		}

		$women_patterns = [
			"/\bwomen\b/i",
			"/\bwomen's\b/i",
			"/\bwomens\b/i",
			"/\bwoman\b/i",
			"/\bfemale\b/i",
			"/\blad(y|ies)\b/i",
			"/\bgirl(s)?\b/i"
		];
		$men_patterns = [
			"/\bmen\b/i",
			"/\bmen's\b/i",
			"/\bmens\b/i",
			"/\bman\b/i",
			"/\bmale\b/i"
		];

		$data['breadcrumbs'] = [];
		$data['breadcrumbs'][] = [
			'text' => $this->language->get('text_home'),
			'href' => $this->url->link('common/home', 'language=' . $this->config->get('config_language'))
		];
		$data['breadcrumbs'][] = [
			'text' => "Women's Collection",
			'href' => $this->url->link('product/women', 'language=' . $this->config->get('config_language'))
		];

		$this->document->setTitle("Women's Collection");

		$all_results = $this->model_catalog_product->getProducts([
			'filter_category_id'  => 492,
			'filter_sub_category' => true,
			'sort'                => $sort,
			'order'               => $order,
			'start'               => 0,
			'limit'               => 1000
		]);

		$filtered = [];

		foreach ($all_results as $product) {
			if (!$this->matchesAnyPattern($product, $women_patterns)) {
				continue;
			}

			if ($this->matchesAnyPattern($product, $men_patterns)) {
				continue;
			}

			if ($selected_tags && !$this->matchesKeywords($product, $selected_tags)) {
				continue;
			}

			if ($selected_brand_ids) {
				$product_brand_ids = $this->getProductBrandCategoryIds((int)$product['product_id'], array_keys($brand_filter_options));

				if (!array_intersect($selected_brand_ids, $product_brand_ids)) {
					continue;
				}
			}

			$filtered[] = $product;
		}

		$product_total = count($filtered);
		$offset = max(0, ($page - 1) * $limit);
		$results = array_slice($filtered, $offset, $limit);

		$data['products'] = [];

		$url = '';

		if (!empty($selected_tags)) {
			$url .= '&women_tags=' . implode(',', $selected_tags);
		}
		if (!empty($selected_brand_ids)) {
			$url .= '&women_brands=' . implode(',', $selected_brand_ids);
		}

		foreach ($results as $result) {
			$description = trim(strip_tags(html_entity_decode($result['description'], ENT_QUOTES, 'UTF-8')));

			if (oc_strlen($description) > $this->config->get('config_product_description_length')) {
				$description = oc_substr($description, 0, $this->config->get('config_product_description_length')) . '..';
			}

			if (!empty($result['image']) && is_file(DIR_IMAGE . html_entity_decode($result['image'], ENT_QUOTES, 'UTF-8'))) {
				$image = $result['image'];
			} else {
				$image = 'placeholder.png';
			}

			if ($this->customer->isLogged() || !$this->config->get('config_customer_price')) {
				$price = $this->currency->format($this->tax->calculate($result['price'], $result['tax_class_id'], $this->config->get('config_tax')), $this->session->data['currency']);
			} else {
				$price = false;
			}

			if ((float)$result['special']) {
				$special = $this->currency->format($this->tax->calculate($result['special'], $result['tax_class_id'], $this->config->get('config_tax')), $this->session->data['currency']);
			} else {
				$special = false;
			}

			if ($this->config->get('config_tax')) {
				$tax = $this->currency->format((float)$result['special'] ? $result['special'] : $result['price'], $this->session->data['currency']);
			} else {
				$tax = false;
			}

			$product_brand_name = '';
			$product_brand_href = '';
			$product_brand_ids = $this->getProductBrandCategoryIds((int)$result['product_id'], array_keys($brand_filter_options));

			if ($product_brand_ids) {
				$first_brand_id = (int)$product_brand_ids[0];

				if (isset($brand_filter_options[$first_brand_id])) {
					$product_brand_name = $brand_filter_options[$first_brand_id];
					$product_brand_href = $this->url->link(
						'product/category',
						'language=' . $this->config->get('config_language') . '&path=492_' . $first_brand_id
					);
				}
			}

			if ($product_brand_name === '' && !empty($result['manufacturer_id'])) {
				$manufacturer_info = $this->model_catalog_manufacturer->getManufacturer((int)$result['manufacturer_id']);

				if ($manufacturer_info) {
					$product_brand_name = (string)$manufacturer_info['name'];
					$product_brand_href = $this->url->link(
						'product/manufacturer.info',
						'language=' . $this->config->get('config_language') . '&manufacturer_id=' . (int)$result['manufacturer_id']
					);
				}
			}

			$product_data = [
				'description' => $description,
				'thumb'       => $this->model_tool_image->resize($image, $this->config->get('config_image_product_width'), $this->config->get('config_image_product_height')),
				'price'       => $price,
				'special'     => $special,
				'tax'         => $tax,
				'minimum'     => $result['minimum'] > 0 ? $result['minimum'] : 1,
				'brand_name'  => $product_brand_name,
				'brand_href'  => $product_brand_href,
				'show_brand_only' => true,
				'href'        => $this->url->link('product/product', 'language=' . $this->config->get('config_language') . '&product_id=' . $result['product_id'] . $url)
			] + $result;

			$data['products'][] = $this->load->controller('product/thumb', $product_data);
		}

		$data['heading_title'] = "Women's Collection";
		$data['text_compare'] = sprintf($this->language->get('text_compare'), isset($this->session->data['compare']) ? count($this->session->data['compare']) : 0);
		$data['compare'] = $this->url->link('product/compare', 'language=' . $this->config->get('config_language'));
		$data['selected_women_tags'] = $selected_tags;
		$data['women_filters'] = $filter_options;
		$data['selected_women_brands'] = $selected_brand_ids;
		$data['women_brand_filters'] = $brand_filter_options;

		$sort_url = '';

		if (!empty($selected_tags)) {
			$sort_url .= '&women_tags=' . implode(',', $selected_tags);
		}
		if (!empty($selected_brand_ids)) {
			$sort_url .= '&women_brands=' . implode(',', $selected_brand_ids);
		}

		$data['sorts'] = [];
		$data['sorts'][] = [
			'text'  => $this->language->get('text_default'),
			'value' => 'p.sort_order-ASC',
			'href'  => $this->url->link('product/women', 'language=' . $this->config->get('config_language') . '&sort=p.sort_order&order=ASC' . $sort_url)
		];
		$data['sorts'][] = [
			'text'  => $this->language->get('text_name_asc'),
			'value' => 'pd.name-ASC',
			'href'  => $this->url->link('product/women', 'language=' . $this->config->get('config_language') . '&sort=pd.name&order=ASC' . $sort_url)
		];
		$data['sorts'][] = [
			'text'  => $this->language->get('text_name_desc'),
			'value' => 'pd.name-DESC',
			'href'  => $this->url->link('product/women', 'language=' . $this->config->get('config_language') . '&sort=pd.name&order=DESC' . $sort_url)
		];
		$data['sorts'][] = [
			'text'  => $this->language->get('text_price_asc'),
			'value' => 'p.price-ASC',
			'href'  => $this->url->link('product/women', 'language=' . $this->config->get('config_language') . '&sort=p.price&order=ASC' . $sort_url)
		];
		$data['sorts'][] = [
			'text'  => $this->language->get('text_price_desc'),
			'value' => 'p.price-DESC',
			'href'  => $this->url->link('product/women', 'language=' . $this->config->get('config_language') . '&sort=p.price&order=DESC' . $sort_url)
		];

		$pagination_url = '';

		if ($sort !== '') {
			$pagination_url .= '&sort=' . $sort;
		}

		if ($order !== '') {
			$pagination_url .= '&order=' . $order;
		}

		if (!empty($selected_tags)) {
			$pagination_url .= '&women_tags=' . implode(',', $selected_tags);
		}
		if (!empty($selected_brand_ids)) {
			$pagination_url .= '&women_brands=' . implode(',', $selected_brand_ids);
		}

		$data['pagination'] = $this->load->controller('common/pagination', [
			'total' => $product_total,
			'page'  => $page,
			'limit' => $limit,
			'url'   => $this->url->link('product/women', 'language=' . $this->config->get('config_language') . $pagination_url . '&page={page}')
		]);

		$data['results'] = sprintf(
			$this->language->get('text_pagination'),
			($product_total) ? (($page - 1) * $limit) + 1 : 0,
			((($page - 1) * $limit) > ($product_total - $limit)) ? $product_total : ((($page - 1) * $limit) + $limit),
			$product_total,
			$limit ? ceil($product_total / $limit) : 1
		);

		$data['sort'] = $sort;
		$data['order'] = $order;
		$data['limit'] = $limit;
		$data['continue'] = $this->url->link('common/home', 'language=' . $this->config->get('config_language'));

		$data['column_left'] = $this->load->controller('common/column_left');
		$data['column_right'] = $this->load->controller('common/column_right');
		$data['content_top'] = $this->load->controller('common/content_top');
		$data['content_bottom'] = $this->load->controller('common/content_bottom');
		$data['footer'] = $this->load->controller('common/footer');
		$data['header'] = $this->load->controller('common/header');

		$this->response->setOutput($this->load->view('product/women', $data));
	}

	/**
	 * Check if product name/description contains any keyword.
	 *
	 * @param array<string, mixed> $product
	 * @param array<int, string> $keywords
	 *
	 * @return bool
	 */
	private function matchesKeywords(array $product, array $keywords): bool {
		$name = oc_strtolower((string)$product['name']);
		$description = oc_strtolower(strip_tags(html_entity_decode((string)$product['description'], ENT_QUOTES, 'UTF-8')));
		$haystack = $name . ' ' . $description;

		foreach ($keywords as $keyword) {
			if (strpos($haystack, oc_strtolower($keyword)) !== false) {
				return true;
			}
		}

		return false;
	}

	/**
	 * Check regex patterns against product name/description.
	 *
	 * @param array<string, mixed> $product
	 * @param array<int, string> $patterns
	 *
	 * @return bool
	 */
	private function matchesAnyPattern(array $product, array $patterns): bool {
		$haystack = (string)$product['name'] . ' ' . strip_tags(html_entity_decode((string)$product['description'], ENT_QUOTES, 'UTF-8'));

		foreach ($patterns as $pattern) {
			if (preg_match($pattern, $haystack)) {
				return true;
			}
		}

		return false;
	}

	/**
	 * @param int $product_id
	 * @param array<int, int> $brandCategoryIds
	 *
	 * @return array<int, int>
	 */
	private function getProductBrandCategoryIds(int $product_id, array $brandCategoryIds): array {
		if (!$brandCategoryIds) {
			return [];
		}

		$id_list = implode(',', array_map('intval', $brandCategoryIds));

		$query = $this->db->query("SELECT DISTINCT cp.path_id AS category_id
			FROM `" . DB_PREFIX . "product_to_category` p2c
			INNER JOIN `" . DB_PREFIX . "category_path` cp ON (cp.category_id = p2c.category_id)
			WHERE p2c.product_id = '" . (int)$product_id . "'
			  AND cp.path_id IN (" . $id_list . ")");

		$ids = [];

		foreach ($query->rows as $row) {
			$ids[] = (int)$row['category_id'];
		}

		return $ids;
	}

	/**
	 * Build brand filter options from direct children of Clothing category.
	 *
	 * @return array<int, string>
	 */
	private function getBrandFilterOptions(): array {
		$options = [];

		$query = $this->db->query("SELECT c.category_id, cd.name
			FROM `" . DB_PREFIX . "category` c
			INNER JOIN `" . DB_PREFIX . "category_description` cd ON (cd.category_id = c.category_id)
			WHERE c.parent_id = '492'
			  AND cd.language_id = '" . (int)$this->config->get('config_language_id') . "'
			ORDER BY cd.name ASC");

		foreach ($query->rows as $row) {
			$options[(int)$row['category_id']] = (string)$row['name'];
		}

		return $options;
	}
}
