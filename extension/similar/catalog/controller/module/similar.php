<?php
namespace Opencart\Catalog\Controller\Extension\Similar\Module;

class Similar extends \Opencart\System\Engine\Controller {
    
public function index(): string {

    if (!$this->config->get('module_similar_status')) {
        return '';
    }

    $this->load->language('extension/similar/module/similar');

    $this->load->model('catalog/product');
    $this->load->model('catalog/category');
    $this->load->model('tool/image');

    if (!isset($this->request->get['product_id'])) {
        return '';
    }

    $product_id = (int)$this->request->get['product_id'];

    // Get categories of current product
    $categories = $this->model_catalog_product->getCategories($product_id);

    $subcategory_id = 0;
    if (!empty($categories)) {
        foreach ($categories as $cat) {
            $category_info = $this->model_catalog_category->getCategory($cat['category_id']);
            if ($category_info && $category_info['parent_id'] != 0) {
                $subcategory_id = (int)$category_info['category_id'];
                break; // take the first subcategory found
            }
        }
    }

    $data['products'] = [];

    if ($subcategory_id) {
        // Search for products inside this subcategory
        $filter_data = [
            'filter_category_id'  => $subcategory_id,
            'filter_sub_category' => false,
            'start'               => 0,
            'limit'               => 10
        ];

        $results = $this->model_catalog_product->getProducts($filter_data);

        foreach ($results as $result) {
            if ($result['product_id'] == $product_id) continue;

            $thumb = '';
            if (!empty($result['image'])) {
                $thumb = $this->model_tool_image->resize($result['image'], 200, 200);
            }

            $product_data = [
                'product_id' => $result['product_id'],
                'thumb'      => $thumb,
                'name'       => $result['name'],
                'price'      => $result['price'],
                'special'    => $result['special'] ?? false,
                'href'       => $this->url->link(
                    'product/product',
                    'product_id=' . $result['product_id']
                )
            ];

            $data['products'][] = $this->load->controller('product/thumb', $product_data);
        }
    }

    // Always render view, even if products array is empty
    $data['heading_title'] = $this->language->get('heading_title');
    $data['text_empty']    = $this->language->get('text_empty');

    return $this->load->view('extension/similar/module/similar', $data);
}



}


 
