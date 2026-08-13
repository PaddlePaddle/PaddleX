from types import SimpleNamespace

from paddlex.inference.common.result.converter import MarkdownConverter


def test_convert_scopes_duplicate_image_paths_by_page():
    image_path = "imgs/img_in_image_box_1_2_3_4.jpg"
    image_0 = object()
    image_1 = object()
    blocks = [
        SimpleNamespace(
            label="image",
            content=f'<img src="{image_path}">',
            image={"path": image_path, "img": image_0},
            page_index=0,
        ),
        SimpleNamespace(
            label="image",
            content=f'<img src="{image_path}">',
            image={"path": image_path, "img": image_1},
            page_index=1,
        ),
    ]

    def page_path(path, source):
        return path.replace("imgs/", f"imgs/page_{source.page_index}_")

    result = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict={"image": lambda block: block.content},
        image_path_transform=page_path,
    )

    assert set(result["markdown_images"]) == {
        "imgs/page_0_img_in_image_box_1_2_3_4.jpg",
        "imgs/page_1_img_in_image_box_1_2_3_4.jpg",
    }
    assert 'src="imgs/page_0_img_in_image_box_1_2_3_4.jpg"' in result["markdown_texts"]
    assert 'src="imgs/page_1_img_in_image_box_1_2_3_4.jpg"' in result["markdown_texts"]


def test_convert_scopes_duplicate_embedded_image_paths_by_page():
    image_path = "imgs/img_in_image_box_1_2_3_4.jpg"
    blocks = [
        SimpleNamespace(
            label="table",
            content=f'<img src="{image_path}">',
            image=None,
            page_index=0,
        ),
        SimpleNamespace(
            label="table",
            content=f'<img src="{image_path}">',
            image=None,
            page_index=1,
        ),
    ]
    imgs_in_doc = [
        {"path": image_path, "img": object(), "page_index": 0},
        {"path": image_path, "img": object(), "page_index": 1},
    ]

    def page_path(path, source):
        page_index = (
            source["page_index"] if isinstance(source, dict) else source.page_index
        )
        return path.replace("imgs/", f"imgs/page_{page_index}_")

    result = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict={"table": lambda block: block.content},
        imgs_in_doc=imgs_in_doc,
        image_path_transform=page_path,
    )

    assert set(result["markdown_images"]) == {
        "imgs/page_0_img_in_image_box_1_2_3_4.jpg",
        "imgs/page_1_img_in_image_box_1_2_3_4.jpg",
    }
    assert 'src="imgs/page_0_img_in_image_box_1_2_3_4.jpg"' in result["markdown_texts"]
    assert 'src="imgs/page_1_img_in_image_box_1_2_3_4.jpg"' in result["markdown_texts"]
