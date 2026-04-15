<?php
namespace Opencart\Catalog\Controller\Common;
/**
 * Class Header
 *
 * Can be called from $this->load->controller('common/header');
 *
 * @package Opencart\Catalog\Controller\Common
 */
class Header extends \Opencart\System\Engine\Controller {
	/**
	 * Index
	 *
	 * @return string
	 */
	public function index(): string {
		// Analytics
		$data['analytics'] = [];

		if (!$this->config->get('config_cookie_id') || (isset($this->request->cookie['policy']) && $this->request->cookie['policy'])) {
			// Extension
			$this->load->model('setting/extension');

			$analytics = $this->model_setting_extension->getExtensionsByType('analytics');

			foreach ($analytics as $analytic) {
				if ($this->config->get('analytics_' . $analytic['code'] . '_status')) {
					$data['analytics'][] = $this->load->controller('extension/' . $analytic['extension'] . '/analytics/' . $analytic['code'], $this->config->get('analytics_' . $analytic['code'] . '_status'));
				}
			}
		}

		$data['lang'] = $this->language->get('code');
		$data['direction'] = $this->language->get('direction');

		$data['title'] = str_ireplace('fashotronic', 'Localspot', (string)$this->document->getTitle());
		$data['base'] = $this->config->get('config_url');
		$data['description'] = $this->document->getDescription();
		$data['keywords'] = $this->document->getKeywords();

		// Hard coding css, so they can be replaced via the event's system.
		$data['bootstrap'] = 'catalog/view/stylesheet/bootstrap.css';
		$data['icons'] = 'catalog/view/stylesheet/fonts/fontawesome/css/all.min.css';
		$data['stylesheet'] = 'catalog/view/stylesheet/stylesheet.css';

		// Hard coding scripts, so they can be replaced via the event's system.
		$data['jquery'] = 'catalog/view/javascript/jquery/jquery-3.7.1.min.js';

		$data['links'] = $this->document->getLinks();
		$data['styles'] = $this->document->getStyles();
		$data['scripts'] = $this->document->getScripts('header');

		// Custom website display name (edit this whenever you want to change site name).
		$data['name'] = 'Localspot';

		// Custom favicon/icon path (edit this whenever you want to change browser icon).
		$custom_icon_path = 'catalog/localspot-removebg-preview.png';

		if (is_file(DIR_IMAGE . $custom_icon_path)) {
			$data['icon'] = $this->config->get('config_url') . 'image/' . $custom_icon_path;
		} else {
			$data['icon'] = '';
		}

		// Custom logo path (edit this line whenever you want to change the logo).
		$custom_logo_path = 'catalog/localspot.jpeg';

		if (is_file(DIR_IMAGE . $custom_logo_path)) {
			$data['logo'] = $this->config->get('config_url') . 'image/' . $custom_logo_path;
		} else {
			$data['logo'] = '';
		}

		$this->load->language('common/header');

		// Wishlist
		if ($this->customer->isLogged()) {
			$this->load->model('account/wishlist');

			$data['text_wishlist'] = sprintf($this->language->get('text_wishlist'), $this->model_account_wishlist->getTotalWishlist($this->customer->getId()));
		} else {
			$data['text_wishlist'] = sprintf($this->language->get('text_wishlist'), (isset($this->session->data['wishlist']) ? count($this->session->data['wishlist']) : 0));
		}

		$data['home'] = $this->url->link('common/home', 'language=' . $this->config->get('config_language'));
		$data['wishlist'] = $this->url->link('account/wishlist', 'language=' . $this->config->get('config_language') . (isset($this->session->data['customer_token']) ? '&customer_token=' . $this->session->data['customer_token'] : ''));
        $data['logged'] = $this->customer->isLogged();
		$data['is_seller'] = $this->customer->isLogged() && ($this->customer->getGroupId() === $this->getSellerGroupId());
		$data['seller_dashboard'] = $this->url->link('account/seller', 'language=' . $this->config->get('config_language') . (isset($this->session->data['customer_token']) ? '&customer_token=' . $this->session->data['customer_token'] : ''));
		$data['account'] = $this->url->link('account/account', 'language=' . $this->config->get('config_language'));
		$data['logout'] = '';

        if (!$this->customer->isLogged()) {
            $data['register'] = $this->url->link('account/register', 'language=' . $this->config->get('config_language'));
            $data['login'] = $this->url->link('account/login', 'language=' . $this->config->get('config_language'));
            $data['firstname'] = ''; // Ensure it's empty for guests
        } else {
            $data['firstname'] = $this->customer->getFirstName(); // Correctly get the name here
            $data['suggestion'] = $this->url->link('account/suggestion', 'language=' . $this->config->get('config_language'));
            $data['account'] = $this->url->link('account/account', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token']);
            $data['order'] = $this->url->link('account/order', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token']);
            $data['transaction'] = $this->url->link('account/transaction', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token']);
            $data['download'] = $this->url->link('account/download', 'language=' . $this->config->get('config_language') . '&customer_token=' . $this->session->data['customer_token']);
            $data['logout'] = $this->url->link('account/logout', 'language=' . $this->config->get('config_language'));
        }

		$data['shopping_cart'] = $this->url->link('checkout/cart', 'language=' . $this->config->get('config_language'));
		$data['checkout'] = $this->url->link('checkout/checkout', 'language=' . $this->config->get('config_language'));
		$data['contact'] = $this->url->link('information/contact', 'language=' . $this->config->get('config_language'));
		$data['telephone'] = $this->config->get('config_telephone');

		$data['language'] = $this->load->controller('common/language');
		$data['currency'] = $this->load->controller('common/currency');
		$data['search'] = $this->load->controller('common/search');
		$data['cart'] = $this->load->controller('common/cart');
		$data['menu'] = $this->load->controller('common/menu');
        $data['route'] = $this->request->get['route'] ?? 'common/home';
		$data['category_clothing'] = $this->url->link('product/category', 'path=492');
        $data['path'] = isset($this->request->get['path']) ? $this->request->get['path'] : '';

        
		return $this->load->view('common/header', $data);
	}

	private function getSellerGroupId(): int {
		$seller_name = 'Seller';
		$language_id = (int)$this->config->get('config_language_id');

		$query = $this->db->query("SELECT customer_group_id FROM `" . DB_PREFIX . "customer_group_description` WHERE language_id = '" . $language_id . "' AND name = '" . $this->db->escape($seller_name) . "' LIMIT 1");

		if ($query->num_rows) {
			return (int)$query->row['customer_group_id'];
		}

		return 0;
	}
}
