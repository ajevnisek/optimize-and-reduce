$(document).ready(function () {
  $('.results-carousel').slick({
    dots: false,
    infinite: false,
    slidesToShow: 4,
    slidesToScroll: 4,
    // autoplay: true,
    // autoplaySpeed: 4000,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          arrows: true,
          slidesToShow: 4,
          slidesToScroll: 4,
        }
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 4,
          slidesToScroll: 3,
        }
      },
      {
        breakpoint: 480,
        settings: {
          slidesToShow: 4,
          slidesToScroll: 4,
        }
      }
    ]
  });
    $('.ood-carousel').slick({
    dots: false,
    infinite: false,
    slidesToShow: 2,
    slidesToScroll: 2,
    // autoplay: true,
    // autoplaySpeed: 4000,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          arrows: true,
          slidesToShow: 2,
          slidesToScroll: 2,
        }
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 2,
        }
      },
      {
        breakpoint: 480,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 2,
        }
      }
    ]
  });
})
